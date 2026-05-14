//int in1 = 0;   // Switch 1 = Pin 12
//int in2 = 1;   // Switch 2 = Pin 11


byte servo_id0 = 13;             // Servo ID0 (head horizontal rotation)
// byte servo_id1 = 13;            // Servo ID1 (right shoulder horizontal rotation)

int rotate_speed = 200;         // One full rotation movement time (seconds × 100) 
int position_offset = 0;        // Overall position adjustment (+ clockwise, - counterclockwise)
int reverse_offset = 0;         // Offset for reverse rotation

int current_position0 = 0;      // Current position of Servo 0 TODO FIX ANGLE MAPPING and add speed to command
int previous_position0 = 1501;  // Previous position of Servo 0


HardwareSerial ServoSerial(1);   // Use UART1
// ESP32-C3 UART pins
#define SERVO_TX 20
#define SERVO_RX 21   // Dummy RX pin (can be unused)

const byte numChar = 32;
char recievedData[numChar];
char tempData[numChar];
bool newData = false;

int id = 0;
int pos = 0;
// int current_position1 = 0;      // Current position of Servo 1
// int previous_position1 = 1501;  // Previous position of Servo 1


byte torque_on_data[] = {0xFA, 0xAF, 0x01, 0x00, 0x24, 0x01, 0x01, 0x01, 0x24}; // Torque ON command
byte position_data[] = {0xFA, 0xAF, 0x01, 0x00, 0x1E, 0x04, 0x01, 0x00, 0x00, 0x64, 0x00, 0x7E};  // Position command

void memcpy(byte* buf, byte* data, int n)     // Data transfer
{
  int sum = 0;
  for(int i = 0; i < n; i++){
    buf[i] = data[i];
  }
}

void checksum(byte* data, int n)     // Set checksum
{
  int sum = 0;
  for(int i = 2; i < n - 1; i++){
    sum = sum ^ data[i];
  }
  data[ n-1 ] = sum;
}

void move_servo(int servo_id, int previous_position, int current_position, bool wait)     // Move servo
{
    byte data[12];
    int move_position;

   // Calculate movement time for this move
    int move_speed = short(abs((long) previous_position - (long) current_position) * (long) rotate_speed / 3600);

    memcpy(data, position_data, 12);         // Copy rotation data

    if (current_position < previous_position) {
      move_position = current_position + position_offset + reverse_offset;
    }
    else {
      move_position = current_position + position_offset;
    }

    memcpy(data, position_data, 12);         // Copy rotation data
    data[2] = servo_id;

    memcpy(&data[7], &move_position, 2);     // Set target position
    memcpy(&data[9], &move_speed, 2);        // Set movement time

    checksum(data, 12);                      // Calculate checksum
    ServoSerial.write(data,12);                   // Rotate to target position

    if (wait) {
      delay(move_speed*10);                  // Wait until rotation is complete
    }
}

void setup() {

 // Enable pull-up resistors for digital inputs
  //pinMode(in1, INPUT_PULLUP);
  // pinMode(in2, INPUT_PULLUP);

  //ServoSerial.begin(115200);    
  Serial.begin(115200);                 // Open serial port at 115200 bps
  ServoSerial.begin(115200, SERIAL_8N1, SERVO_RX, SERVO_TX);
  delay(500);                               // Wait 0.5 seconds

  torque_on_data[2] = 13;            // Servo ID0
  checksum(torque_on_data, 9);              // Calculate checksum
  ServoSerial.write(torque_on_data,9);           // Turn servo torque ON

  torque_on_data[2] = 14;            // Servo ID1
  checksum(torque_on_data, 9);              // Calculate checksum
  ServoSerial.write(torque_on_data,9);           // Turn servo torque ON

  torque_on_data[2] = 15;            // Servo ID1
  checksum(torque_on_data, 9);              // Calculate checksum
  ServoSerial.write(torque_on_data,9);

  delay(500);                               // Wait 0.5 seconds

  current_position0 = 0;                    // Servo 0 home position
  move_servo(servo_id0, previous_position0, current_position0, false);
  previous_position0 = current_position0;

//   current_position1 = 600;                  // Servo 1 home position
//   move_servo(servo_id1, previous_position1, current_position1, false);
//   previous_position1 = current_position1;

//  // current_position2 = 400;                // Servo 2 home position (Robi 1)
//   current_position2 = 250;                  // Servo 2 home position (Robi 2)
//   move_servo(servo_id2, previous_position2, current_position2, false);
//   previous_position2 = current_position2;

//   current_position3 = -600;                 // Servo 3 home position
//   move_servo(servo_id3, previous_position3, current_position3, false);
//   previous_position3 = current_position3;

//  // current_position4 = -400;               // Servo 4 home position (Robi 1)
//   current_position4 = -250;                 // Servo 4 home position (Robi 2)
//   move_servo(servo_id4, previous_position4, current_position4, false);
//   previous_position4 = current_position4;

  //rotate_speed = 1000;                      // Test speed
  delay(1000);                              // Wait 1 second
}

void loop() {
  recvWithEndMarker();
  if (newData) {
    strcpy(tempData, recievedData); // Copy to temp for safe parsing
    parseData();
    servo_id0 = id;
    delay(500); 
    torque_on_data[2] = servo_id0;            // Servo ID0
    checksum(torque_on_data, 9);              // Calculate checksum
    ServoSerial.write(torque_on_data,9); 
    delay(500); 

    current_position0 = pos;                  // Servo 0
    move_servo(servo_id0, previous_position0, current_position0, false);
    previous_position0 = current_position0;

    newData = false;
  }
}

void recvWithEndMarker() {
  static byte ndx = 0;
  char endMarker = '\n';
  char rc;
  
  while (Serial.available() > 0 && newData == false) {
    rc = Serial.read();
    if (rc != endMarker) {
      if (ndx < numChar - 1) {
        recievedData[ndx] = rc;
        ndx++;
      }
    } else {
      recievedData[ndx] = '\0'; // Terminate string
      ndx = 0;
      newData = true;
    }
  }
}

void parseData() {
  char *strIndx;
  
  // Split by space (or use "," for comma-separated)
  strIndx = strtok(tempData, ":"); 
  id = atoi(strIndx); // Convert first part to int
  
  strIndx = strtok(NULL, ":");
  pos = atoi(strIndx); // Convert second part to int
}