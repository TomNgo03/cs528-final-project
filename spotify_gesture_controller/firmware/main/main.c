#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "driver/i2c.h"
#include "esp_err.h"
#include "esp_check.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "mpu6050.h"

static const char *TAG = "SpotifyGesture";

#define I2C_MASTER_SCL_IO           1
#define I2C_MASTER_SDA_IO           0
#define I2C_MASTER_NUM              I2C_NUM_0
#define I2C_MASTER_FREQ_HZ          100000
#define I2C_MASTER_TX_BUF_DISABLE   0
#define I2C_MASTER_RX_BUF_DISABLE   0

#define MPU6050_ADDR                0x68
#define SAMPLE_RATE_HZ              100
#define SAMPLE_PERIOD_MS            (1000 / SAMPLE_RATE_HZ)
#define COLLECTION_DURATION_MS      3000
#define SAMPLES_PER_GESTURE         (SAMPLE_RATE_HZ * COLLECTION_DURATION_MS / 1000)
#define COMMAND_BUF_SIZE            256

typedef struct {
    const char *name;
    uint8_t address;
    mpu6050_handle_t handle;
} imu_sensor_t;

typedef enum {
    MODE_COLLECT = 0,
    MODE_STREAM
} app_mode_t;

static imu_sensor_t sensors[] = {
    {.name = "hand_imu", .address = MPU6050_ADDR, .handle = NULL},
};

static const size_t sensor_count = sizeof(sensors) / sizeof(sensors[0]);
static char current_gesture[32] = "tap_index";
static app_mode_t current_mode = MODE_COLLECT;

static esp_err_t i2c_master_init(void)
{
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = I2C_MASTER_SDA_IO,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_io_num = I2C_MASTER_SCL_IO,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_MASTER_FREQ_HZ,
    };

    ESP_RETURN_ON_ERROR(i2c_param_config(I2C_MASTER_NUM, &conf), TAG, "I2C config failed");
    ESP_RETURN_ON_ERROR(i2c_driver_install(I2C_MASTER_NUM, conf.mode, I2C_MASTER_RX_BUF_DISABLE,
                                           I2C_MASTER_TX_BUF_DISABLE, 0), TAG, "I2C driver install failed");
    ESP_LOGI(TAG, "I2C initialized: SDA GPIO %d, SCL GPIO %d, %d Hz",
             I2C_MASTER_SDA_IO, I2C_MASTER_SCL_IO, I2C_MASTER_FREQ_HZ);
    return ESP_OK;
}

static esp_err_t imu_init_all(void)
{
    for (size_t i = 0; i < sensor_count; i++) {
        sensors[i].handle = mpu6050_create(I2C_MASTER_NUM, sensors[i].address);
        if (sensors[i].handle == NULL) {
            ESP_LOGE(TAG, "Failed to create MPU6050 handle for %s", sensors[i].name);
            return ESP_FAIL;
        }

        ESP_RETURN_ON_ERROR(mpu6050_config(sensors[i].handle, ACCE_FS_4G, GYRO_FS_500DPS),
                            TAG, "MPU6050 config failed");

        uint8_t whoami = 0;
        if (mpu6050_get_deviceid(sensors[i].handle, &whoami) == ESP_OK) {
            ESP_LOGI(TAG, "%s WHO_AM_I: 0x%02X", sensors[i].name, whoami);
        }
    }
    return ESP_OK;
}

static bool read_primary_imu(mpu6050_acce_value_t *acce, mpu6050_gyro_value_t *gyro)
{
    if (sensor_count == 0 || sensors[0].handle == NULL) {
        return false;
    }
    esp_err_t ret = mpu6050_get_acce(sensors[0].handle, acce);
    if (ret == ESP_OK) {
        ret = mpu6050_get_gyro(sensors[0].handle, gyro);
    }
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "IMU read failed: %s", esp_err_to_name(ret));
        return false;
    }
    return true;
}

static void print_csv_header(void)
{
    printf("Time(ms),AccelX(g),AccelY(g),AccelZ(g),GyroX(dps),GyroY(dps),GyroZ(dps)\n");
}

static void print_imu_row(int64_t elapsed_ms, const mpu6050_acce_value_t *acce,
                          const mpu6050_gyro_value_t *gyro)
{
    printf("%lld,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
           elapsed_ms,
           acce->acce_x, acce->acce_y, acce->acce_z,
           gyro->gyro_x, gyro->gyro_y, gyro->gyro_z);
}

static void collect_gesture_window(const char *gesture)
{
    printf("\n========== START_GESTURE_%s ==========\n", gesture);
    print_csv_header();

    const int64_t start_ms = esp_timer_get_time() / 1000;
    int samples = 0;
    while (samples < SAMPLES_PER_GESTURE) {
        mpu6050_acce_value_t acce;
        mpu6050_gyro_value_t gyro;
        if (read_primary_imu(&acce, &gyro)) {
            int64_t elapsed_ms = (esp_timer_get_time() / 1000) - start_ms;
            print_imu_row(elapsed_ms, &acce, &gyro);
            samples++;
        }
        vTaskDelay(pdMS_TO_TICKS(SAMPLE_PERIOD_MS));
    }

    printf("========== END_GESTURE_%s ==========\n", gesture);
}

static void stream_imu_forever(void)
{
    ESP_LOGI(TAG, "Real-time streaming mode. Python can read sliding windows from CSV rows.");
    print_csv_header();
    const int64_t start_ms = esp_timer_get_time() / 1000;
    while (current_mode == MODE_STREAM) {
        mpu6050_acce_value_t acce;
        mpu6050_gyro_value_t gyro;
        if (read_primary_imu(&acce, &gyro)) {
            int64_t elapsed_ms = (esp_timer_get_time() / 1000) - start_ms;
            print_imu_row(elapsed_ms, &acce, &gyro);
        }
        vTaskDelay(pdMS_TO_TICKS(SAMPLE_PERIOD_MS));
    }
}

static void print_help(void)
{
    ESP_LOGI(TAG, "Commands over serial:");
    ESP_LOGI(TAG, "  label <gesture_name>  set collection label");
    ESP_LOGI(TAG, "  collect               record one 3 second labeled window");
    ESP_LOGI(TAG, "  auto                  repeatedly collect 3 second windows");
    ESP_LOGI(TAG, "  stream                real-time continuous CSV mode");
    ESP_LOGI(TAG, "  help                  show commands");
}

static void handle_command(char *line)
{
    char *newline = strchr(line, '\n');
    if (newline) {
        *newline = '\0';
    }
    char *carriage = strchr(line, '\r');
    if (carriage) {
        *carriage = '\0';
    }

    if (strncmp(line, "label ", 6) == 0) {
        strncpy(current_gesture, line + 6, sizeof(current_gesture) - 1);
        current_gesture[sizeof(current_gesture) - 1] = '\0';
        ESP_LOGI(TAG, "Current gesture label: %s", current_gesture);
    } else if (strcmp(line, "collect") == 0) {
        current_mode = MODE_COLLECT;
        collect_gesture_window(current_gesture);
    } else if (strcmp(line, "auto") == 0) {
        current_mode = MODE_COLLECT;
        ESP_LOGI(TAG, "Auto collection for label %s. Reset board to stop.", current_gesture);
        while (current_mode == MODE_COLLECT) {
            collect_gesture_window(current_gesture);
            vTaskDelay(pdMS_TO_TICKS(1000));
        }
    } else if (strcmp(line, "stream") == 0) {
        current_mode = MODE_STREAM;
        stream_imu_forever();
    } else {
        print_help();
    }
}

void app_main(void)
{
    ESP_LOGI(TAG, "Invisible Spotify Controller firmware");
    ESP_LOGI(TAG, "Sample rate: %d Hz, window: %d ms, samples/window: %d",
             SAMPLE_RATE_HZ, COLLECTION_DURATION_MS, SAMPLES_PER_GESTURE);

    ESP_ERROR_CHECK(i2c_master_init());
    ESP_ERROR_CHECK(imu_init_all());
    print_help();

    char line[COMMAND_BUF_SIZE];
    while (true) {
        ESP_LOGI(TAG, "Waiting for command. Default label is %s", current_gesture);
        if (fgets(line, sizeof(line), stdin) != NULL) {
            handle_command(line);
        }
    }
}
