//SERVO BUS CONTROLLER FOR DEAGNOSTINI ROBI ROBOT.
//USING ESP32 C3 MINI, BIDIRECTIONAL LOGIC LEVEL SHIFTER, EXTERNAL 5V POWER SOURCE.
//RECIEVES COMMANDS FROM HOST APPLICATION AND PARSES THEM INTO A VALID SERVO PACKET.

//BUS 1 PINS
#define SERVO_TX 20             //NOTE: RX & TX PIN (HALF DUPLEX)
#define SERVO_RX 21             //NOTE: NOT USED
HardwareSerial ServoSerial(1);  //NOTE: UART1/BUS1

byte servo_id = 0;              //NOTE: SERVO ID BYTE, DEFAULT IS "0"
int rotate_speed = 200;         //NOTE: FULL ROTATION TIME (SECONDS X 100)
int position_offset = 0;        //NOTE: POSITION ADJUSTMENT (+ CLOCKWISE, - COUNTERCLOCKWISE)
int reverse_offset = 0;         //NOTE: REVERSE ROTATION OFFSET
int current_position0 = 0;      //NOTE: CURRENT/NEW POSITION
int previous_position0 = 1501;  //NOTE: LAST POSITION
int min_Servo_Range = 0;        //NOTE: SERVO MIN RANGE
int max_Servo_Range = 1501;     //NOTE: SERVO MAX RANGE

const byte numChar = 32;        //NOTE: SET MAX LEGNTH OF COMMAND PACKET
char recievedData[numChar];     //NOTE: DATA RECIEVED FROM HOST
char tempData[numChar];         //NOTE: TEMP TO STORE DATA SAFELY FOR PROCESSING
bool newData = false;           //NOTE: SETS IF A NEW COMMAND CAN BE READ AND SENT

int id = 0;                     //NOTE: SELECTED ID RECEIVED FROM HOST
int pos = 0;                    //NOTE: ANGLE RECEIVED FROM HOST

byte torque_on_data[] = {0xFA, 0xAF, 0x01, 0x00, 0x24, 0x01, 0x01, 0x01, 0x24};                   //NOTE: TORQUE ON COMMAND BYTES
byte position_data[] = {0xFA, 0xAF, 0x01, 0x00, 0x1E, 0x04, 0x01, 0x00, 0x00, 0x64, 0x00, 0x7E};  //NOTE: POSITION COMMAND BYTES

//INITIALISE SERVOS AND SERIAL COMMUNICATIONS
void setup() 
{
  Serial.begin(115200);                     //NOTE: OPEN COMMUNICATION TO HOST
  ServoSerial.begin(115200, SERIAL_8N1, SERVO_RX, SERVO_TX); //NOTE: SET SERIAL PIN & OPEN COMMUNICATION TO SERVO BUS 1 (HEAD)
  delay(500);                               //NOTE: ALLOW TIME FOR SERIAL INIT

//NOTE: REDUNDANT
  torque_on_data[2] = servo_id;             //NOTE: CALCULATE TORQUE ON DATA FOR SERVO
  checksum(torque_on_data, 9);              //NOTE: CALCULATE SERVO COMMAND CHECKSUM
  ServoSerial.write(torque_on_data,9);      //NOTE: SET TORQUE ON FOR SELECTED SERVO

//NOTE: REDUNDANT
  torque_on_data[2] = 15;                   //NOTE: CALCULATE TORQUE ON DATA FOR SERVO
  checksum(torque_on_data, 9);              //NOTE: CALCULATE SERVO COMMAND CHECKSUM
  ServoSerial.write(torque_on_data,9);      //NOTE: SET TORQUE ON FOR SELECTED SERVO

  delay(500);                               //NOTE: ALLOW TIME FOR SERVO INIT

  current_position0 = 0;                    //NOTE: HOME THE SERVO (MUST SETUP FOR ALL SERVOS OR SWAP TO MOTION FILES SUCH AS REST or STAND)
  move_servo(servo_id, previous_position0, current_position0, false);
  previous_position0 = current_position0;

  //rotate_speed = 1000;                    //NOTE: TESTING SPEED, HIGHER VALUES RESULT IN SLOWER MOVEMENT
  delay(1000);                              //NOTE: ALLOW TIME TO FINISH STARTUP
}



//DETECT SERIAL AND RECIEVE COMMANDS
void loop() 
{
  recvWithEndMarker();                        //NOTE RECIEVES COMMANDS FROM HOST
  if (newData) 
  {
    strcpy(tempData, recievedData);           //NOTE: COPY TO TEMP VAR FOR SAFETY
    parseData();
    servo_id = id;                            //NOTE: SET SERVO ID FROM HOST COMMAND
    torque_on_data[2] = servo_id;             //NOTE: CALCULATE TORQUE ON DATA FOR SERVO
    checksum(torque_on_data, 9);              //NOTE: CALCULATE SERVO COMMAND CHECKSUM
    ServoSerial.write(torque_on_data,9);      //NOTE: SET TORQUE ON FOR SELECTED SERVO

    current_position0 = pos;                  //TODO: CONVERT NAMES AND EDIT LOGIC TO MAP RANGE FROM 0-180 to 0,1501
    move_servo(servo_id, previous_position0, current_position0, false);
    previous_position0 = current_position0;

    newData = false;                          //NOTE: CAN NOW RECEIVE FRESH SERVO COMMAND
  }
}



//DATA TRANSFER
void memcpy(byte* buf, byte* data, int n)    //NOTE: NEED TO VALIDATE
{
  int sum = 0;
  for(int i = 0; i < n; i++)
  {
    buf[i] = data[i];
  }
}



//SET CHECKSUM
void checksum(byte* data, int n)     //NOTE: NEED TO VALIDATE
{
  int sum = 0;
  for(int i = 2; i < n - 1; i++)
  {
    sum = sum ^ data[i];
  }
  data[ n-1 ] = sum;
}



//MOVE THE SERVO
void move_servo(int servo_id, int previous_position, int current_position, bool wait) 
{
    byte data[12];
    int move_position;
    int move_speed = short(abs((long) previous_position - (long) current_position) * (long) rotate_speed / 3600); //NOTE: CALCULATE TIME FOR MOVEMENT TO FINISH

    memcpy(data, position_data, 12);         //NOTE: COPY ROTATION DATA

    if (current_position < previous_position) 
    {
      move_position = current_position + position_offset + reverse_offset;
    }
    else 
    {
      move_position = current_position + position_offset;
    }

    memcpy(data, position_data, 12);         //NOTE: COPY ROTATION DATA
    data[2] = servo_id;

    memcpy(&data[7], &move_position, 2);     //NOTE: SET TARGET POSITION
    memcpy(&data[9], &move_speed, 2);        //NOTE: SET MOVE SPEED (BASED ON MOVE TIME)

    checksum(data, 12);                      //NOTE: CALCULATE THE SERVO COMMAND CHECKSUM
    ServoSerial.write(data,12);              //NOTE: SEND COMMAND TO SELECTED SERVO

    if (wait) 
    {
      delay(move_speed*10);                  //NOTE: WAIT UNTIL MOTION IS COMPLETE
    }
}



//GET COMMAND FROM SERIAL AFTER END BYTE
void recvWithEndMarker() 
{
  static byte ndx = 0;
  char endMarker = '\n';
  char rc;
  
  while (Serial.available() > 0 && newData == false) 
  {
    rc = Serial.read();           //NOTE: READ COMMAND FROM HOST
    if (rc != endMarker) 
    {
      if (ndx < numChar - 1) 
      {
        recievedData[ndx] = rc;
        ndx++;
      }
    } 
    else 
    {
      recievedData[ndx] = '\0';   //NOTE: TERMINATE STRING
      ndx = 0;
      newData = true;            
    }
  }
}



//PARSE COMMAND INTO VALID SERVO DATA
void parseData() 
{
  char *strIndx;
  
  //NOTE: DATA DIVIDER is ":" (EXAMPLE "ID:ANGLE")
  strIndx = strtok(tempData, ":");  //NOTE: DATA DIVIDER TO SEPERATE COMMAND INTO COMPONENTS
  id = atoi(strIndx);               //NOTE: CONVERT FIRST BYTES TO VALID SERVO ID
  
  strIndx = strtok(NULL, ":");      //NOTE: DATA DIVIDER TO SEPERATE COMMAND INTO COMPONENTS
  pos = atoi(strIndx);              //NOTE: CONVERT LAST BYTES TO VALID SERVO ANGLE
}