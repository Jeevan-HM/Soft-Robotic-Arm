#include <Ethernet.h>
#include <SPI.h>
#include <Adafruit_MCP4725.h>
Adafruit_MCP4725 dac;
#include <Adafruit_ADS1X15.h>
Adafruit_ADS1115 adc;
#include <Wire.h>

//Arduino Analog Vars
int Analog0 = A0; //Pressure Sensor
int Analog1 = A1;
int Analog2 = A2;
int Analog3 = A3; //Reading Analog Voltage Input to Regulator
int Analog4 = A4;
int Analog5 = A5; //Reading Pressure from Pressure Regulator

//ADC Analog Vars
int16_t adc0, adc1, adc2, adc3;

float Bit0;
float Bit1;
float Bit2;
float Bit3;
float Bit4;
float Bit5;

float pressure;
float pressure_volt;
float pres_reg_volt;
float set_pressure;
float volt;

// Change to a unique MAC address
//byte mac[] = {0x/DE, 0xAD, 0xBE, 0xFE , 0xFE, 0xED}; //ARD 1
byte mac[] = {0x02, 0x4E, 0xA3, 0x56 , 0x9F, 0xB2}; //ARD 2
//byte mac[] = {0xBE, 0xDE, 0xBE, 0xED , 0xFE, 0xED}; //ARD 3
//byte mac[] = {0xBE, 0xDE, 0xBE, 0xED , 0xFE, 0xFE}; //ARD 4
//byte mac[] = {0xBE, /0xDE, 0xBE, 0xED , 0xED, 0xED}; //ARD 5
//byte mac[] = {0x0A, 0x1A, 0x2C, 0x3B, 0x4D, 0x5E}; //ARD 6
//byte mac[] = {0x0A, 0x2B, 0x3C, 0x4D, 0x5E, 0x6F}; //ARD 7
//byte mac[] = {0x0A, 0x1D, 0x2E, 0x3F, 0x4A, 0x5B}; //ARD 8
//byte mac[] = {0x0A, 0x1E, 0x2F, 0x3A, 0x4B, 0x5C}; //ARD 9
//byte mac[] = {0x0A, 0x1F, 0x2A, 0x3B, 0x4C, 0x5D}; //ARD 10
//byte mac[] = {0x0A, 0x1A, 0x2B, 0x3C, 0x4D, 0x5E}; //ARD 11

char hostname[] = "arduino"; // Set a hostname for the Arduino
byte ip[] = { 10,211,215,21}; //set unique Arduino ip addr #1 (ex: Ard 2 = 21)
byte subnet[] = { 255, 255, 0, 0 };
//byte pcIP[] = { 10,203,49,197 }; //Replace with PC IP 10,203,49,197
byte pcIP[] = {10,211,215,251};
int port = 10002; //unique ard port num

EthernetServer server(6666);
EthernetClient client;
float sensorValue;

void setup() {
  Ethernet.begin(mac, ip, subnet);
  server.begin();
  Wire.begin();
  Serial.begin(9600);
  Serial.println("Starting DAC");
  adc.setGain(GAIN_TWOTHIRDS);
  adc.begin(); //automatically uses 0x48 as ADC ADDR
  dac.begin(0x62);
  dac.setVoltage(0,false);
  connectToPC();
}

void connectToPC() {
  while (!client.connected()) {
    Serial.println("Connecting to PC...");
    if (client.connect(pcIP, port)) {
      Serial.println("Connected to PC");
    }
    else {
      Serial.println("Connection to PC failed, closing regulator.., retrying in 1 second...");
      dac.setVoltage(0,false);
      delay(1000); // Wait for 1 seconds before attempting to reconnect
    }
}
}  


void loop() {
  
float received_float = 0.0;

  // Check if the connection is still alive
  if (!client.connected()) {
    Serial.println("Connection lost, attempting to reconnect...");
    dac.setVoltage(0,false);
    connectToPC();  // Reconnect if the connection is lost
  }

  // Check for incoming data from PC
//  while (!client.available()){
  //  Serial.println("Not getting data from PC");
  //}

  while (client.available()) {
    // Read 4 bytes for the float
    uint8_t received_data[4];
    for (int i = 0; i < 4; i++) {
      received_data[i] = client.read();
    }

    // Unpack the received float
    received_float = *((float *)received_data);
    
    set_pressure = 20.0;



    // Reading from all Ard Analog ports
    Bit0 = analogRead(Analog0);
    Bit1 = analogRead(Analog1);
    Bit2 = analogRead(Analog2);
    Bit3 = analogRead(Analog3);
    Bit4 = analogRead(Analog4);
    Bit5 = analogRead(Analog5);
    
    // Reading from all ADC Analog ports
    adc0 = adc.readADC_SingleEnded(0);
    adc1 = adc.readADC_SingleEnded(1);
    adc2 = adc.readADC_SingleEnded(2);
    adc3 = adc.readADC_SingleEnded(3);

    int counter = int((received_float/set_pressure)*4095);
    dac.setVoltage(counter, false);
    volt = ((Bit3/1023.0)*5.0);
    //Serial.print("Received data: ");
    // Serial.print("Desired Pressure: ");
    // Serial.print(received_float);
    // Serial.print("  ,  ");
    // Serial.print("Regulator Voltage: ");
    //Serial.print(volt);
    //Serial.print("  ,  ");
  
    //Calculating Pressure from pressure sensor
    pressure_volt = Bit0*(5.0/1023.0);
    pressure = ((20.0-0.0)*(pressure_volt-(0.1*5.0)))/(0.8*5.0);

    uint8_t packed_data[12];
    
    //int16_t sensor_value[6] = {(int16_t)Bit0, (int16_t)Bit1, (int16_t)Bit2, (int16_t)Bit3, (int16_t)Bit4, (int16_t)Bit5};
    int16_t sensor_value[4] = {(int16_t)adc0, (int16_t)adc1, (int16_t)adc2, (int16_t)adc3};


    //Serial.print("Pressure Sensor Bit: ");
    //Serial.print(sensor_value[0]);
    //Serial.print("  ,  ");
    //Serial.print(pressure);
    //Serial.print("psi");
    //Serial.print("\n");

  
    /* Pack the data into bytes
    for (int i = 0; i < 6; i++) {
      packed_data[2 * i] = (sensor_value[i] >> 8) & 0xFF;      // High byte of the ith integer
      packed_data[2 * i + 1] = sensor_value[i] & 0xFF;         // Low byte of the ith integer
    } */

    // Pack the data into bytes
    for (int i = 0; i < 4; i++) {
      packed_data[2 * i] = (sensor_value[i] >> 8) & 0xFF;      // High byte of the ith integer
      packed_data[2 * i + 1] = sensor_value[i] & 0xFF;         // Low byte of the ith integer
    }
    
    /* Send the data
    for (int i = 0; i < 12; i++) {
      client.write(packed_data[i]);
      //delay(0.1);
      } */

      // Send the data
    for (int i = 0; i < 8; i++) {
      client.write(packed_data[i]);
      //delay(0.1);
      }
    
    

}

}
