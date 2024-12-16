#include <Arduino_LSM6DS3.h>
#include <DHT.h>
#include <WiFiNINA.h>
#include <WiFiClient.h>

const char* ssid = "Predator";
const char* password = "123456789";
const char* serverIP = "192.168.137.107";  // IP address of the Raspberry Pi

#define DHTPIN 2
#define DHTTYPE DHT22
#define BUZZER_PIN 3
#define LED_PIN 4

DHT dht(DHTPIN, DHTTYPE);
WiFiServer server(5000);
WiFiClient client;  // WiFi client for making HTTP requests

float ax, ay, az;
float gx, gy, gz;
bool tsunamiAlert = false;
bool earthquakeAlert = false;
unsigned long alertStartTime = 0;
const unsigned long BUZZER_DURATION = 5000;  // Duration for buzzer and LED activation
const unsigned long debounceDuration = 3000;  // Duration to debounce alerts
unsigned long lastEarthquakeAlert = 0;
unsigned long lastTsunamiAlert = 0;

void sendResponse(WiFiClient& client, const char* message, int statusCode = 200) {
    client.print("HTTP/1.1 ");
    client.print(statusCode);
    client.println(" OK");
    client.println("Content-Type: text/plain");
    client.println();
    client.println(message);
}

void setup() {
    Serial.begin(115200);
    dht.begin();
    pinMode(BUZZER_PIN, OUTPUT);
    digitalWrite(BUZZER_PIN, LOW);
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    if (!IMU.begin()) {
        Serial.println("Failed to initialize IMU!");
        while (1);
    } else {
        Serial.println("IMU initialized successfully.");
    }

    Serial.print("Connecting to Wi-Fi...");
    int connectionAttempts = 0;
    while (WiFi.begin(ssid, password) != WL_CONNECTED && connectionAttempts < 10) {
        delay(1000);
        Serial.print(".");
        connectionAttempts++;
    }
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\nConnected to Wi-Fi!");
        Serial.print("IP Address: ");
        Serial.println(WiFi.localIP());
    } else {
        Serial.println("\nFailed to connect to Wi-Fi.");
    }

    server.begin();
}

void loop() {
    readSensors();
    handleAlerts();

    WiFiClient client = server.available();
    if (client) {
        Serial.println("Client connected!");
        String request = client.readStringUntil('\r');
        client.flush();

        if (request.indexOf("GET /ACTIVATE_BUZZER_LED") != -1) {
            digitalWrite(BUZZER_PIN, HIGH);
            digitalWrite(LED_PIN, HIGH);
            sendResponse(client, "Buzzer and LED activated.");
        } else if (request.indexOf("GET /DEACTIVATE_BUZZER_LED") != -1) {
            digitalWrite(BUZZER_PIN, LOW);
            digitalWrite(LED_PIN, LOW);
            sendResponse(client, "Buzzer and LED deactivated.");
        } else {
            sendResponse(client, "Command not recognized.", 404);
        }

        client.stop();
        Serial.println("Client disconnected.");
    }

    delay(1000);
}

void readSensors() {
    float temperature = dht.readTemperature();
    float humidity = dht.readHumidity();

    if (!isnan(temperature) && !isnan(humidity)) {
        Serial.print("Temperature: ");
        Serial.print(temperature);
        Serial.print(" °C, Humidity: ");
        Serial.print(humidity);
        Serial.println(" %");
    } else {
        Serial.println("Failed to read temperature or humidity.");
    }

    if (IMU.accelerationAvailable()) {
        IMU.readAcceleration(ax, ay, az);
        if (abs(ax) > 2.0 || abs(ay) > 2.0 || abs(az) > 2.0) {
            if (millis() - lastTsunamiAlert > debounceDuration) {
                tsunamiAlert = true;
                lastTsunamiAlert = millis();
                alertStartTime = millis();
                Serial.println("Warning: Sudden acceleration detected! Possible tsunami alert!");
                sendAlertToPi("tsunami");
            }
        }
    }

    if (IMU.gyroscopeAvailable()) {
        IMU.readGyroscope(gx, gy, gz);
        if (abs(gx) > 3.0 || abs(gy) > 3.0 || abs(gz) > 3.0) {
            if (millis() - lastEarthquakeAlert > debounceDuration) {
                earthquakeAlert = true;
                lastEarthquakeAlert = millis();
                alertStartTime = millis();
                Serial.println("Warning: Sudden rotation detected! Possible earthquake alert!");
                sendAlertToPi("earthquake");
            }
        }
    }
}

void handleAlerts() {
    if (tsunamiAlert || earthquakeAlert) {
        digitalWrite(BUZZER_PIN, HIGH);
        digitalWrite(LED_PIN, HIGH);

        if (millis() - alertStartTime >= BUZZER_DURATION) {
            tsunamiAlert = false;
            earthquakeAlert = false;

            digitalWrite(BUZZER_PIN, LOW);
            digitalWrite(LED_PIN, LOW);
            Serial.println("Alert duration ended.");
        }
    }
}

void sendAlertToPi(const char* alertType) {
    if (client.connect(serverIP, 5000)) {
        client.print("POST /alert HTTP/1.1\r\n");
        client.print("Host: ");
        client.print(serverIP);
        client.print("\r\n");
        client.print("Content-Type: application/x-www-form-urlencoded\r\n");
        client.print("Content-Length: ");
        client.print(strlen(alertType) + 6);  // "type=" + alertType length
        client.print("\r\n\r\n");
        client.print("type=");
        client.print(alertType);
        client.print("\r\n");
        client.stop();
    }
}

void sendAlert(const char* alertType) {
    Serial.println(alertType);
}
