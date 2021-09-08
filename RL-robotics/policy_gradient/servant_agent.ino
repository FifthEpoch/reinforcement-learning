#include <DallasTemperature.h>
#include <OneWire.h>
#include <Servo.h>

bool debug_mode = true;

/*   SET UP ----------------------------------------------------------------------------
 *   Left motor is referred to as A
 *   Right motor is referred to as B 
*/
const int A_ENABLE = 6; 
const int A0_IN = 9;
const int A1_IN = 10;

const int B_ENABLE = 5;
const int B0_IN = 7;
const int B1_IN = 8;

/*  Bump sensors 
*/
const int l_SWITCH_PIN = A2;
const int r_SWITCH_PIN = A3;

/*  Ultrasonic sensor & servo 
*/
Servo head;
const int Echo_PIN = 2;
const int Trig_PIN = 3;
const int SERVO_PIN = 11;
const int BUZZ_PIN = 13;
const int distancelimit = 20; //distance limit for obstacles in front           
const int sidedistancelimit = 20; //minimum distance in cm to obstacles at both sides (the car will allow a shorter distance sideways)
int distance;

/*  Photo transistors 
*/
const int l_PHOTO_PIN = A0;
const int r_PHOTO_PIN = A5;

/*  Temp sensor
*/
const int TEMP_PIN = 4;
OneWire oneWire(TEMP_PIN);
DallasTemperature sensors(&oneWire);


void setup() {
  stop_motors();
  
  Serial.begin(115200);

  // blink 3 times
  pinMode(LED_BUILTIN, OUTPUT);
  for (int i = 0; i < 3; i++) {
    blink_internal_led();
  }

  // set pin mode
  pinMode(A_ENABLE, OUTPUT);
  pinMode(A0_IN, OUTPUT);
  pinMode(A1_IN, OUTPUT);

  pinMode(B_ENABLE, OUTPUT);
  pinMode(B0_IN, OUTPUT);
  pinMode(B1_IN, OUTPUT);

  pinMode(l_SWITCH_PIN, INPUT);
  pinMode(r_SWITCH_PIN, INPUT);

  pinMode(Trig_PIN, OUTPUT); 
  pinMode(Echo_PIN,INPUT); 

  pinMode(TEMP_PIN, INPUT);

  /*init buzzer*/
  digitalWrite(Trig_PIN,LOW);
  
  /*init servo*/
  head.attach(SERVO_PIN); 
  head.write(90);
}

void blink_internal_led() 
{  
  digitalWrite(LED_BUILTIN, HIGH);
  delay(100);
  digitalWrite(LED_BUILTIN, LOW);
  delay(100);
}

void set_dir(bool _m_left, bool _m_right) {

  bool m_l_upper = (_m_left) ? LOW : HIGH;
  bool m_l_lower = (_m_left) ? HIGH : LOW;

  bool m_r_upper = (_m_right) ? LOW : HIGH;
  bool m_r_lower = (_m_right) ? HIGH : LOW;

  digitalWrite(A0_IN, m_l_upper);
  digitalWrite(A1_IN, m_l_lower);
  
  digitalWrite(B0_IN, m_r_upper);
  digitalWrite(B1_IN, m_r_lower);
}

void set_speed(int _l_speed, int _r_speed) {
  
  if (_l_speed == _r_speed)
  {
    for (int i = 0; i <= _r_speed; i++) 
    {
        analogWrite(A_ENABLE, i + 7);
        analogWrite(B_ENABLE, i);  // right wheel needs more power to start
        delay(8);
    }
  } else 
  {
    int faster_motor = (_l_speed >= _r_speed) ? 0 : 1;
    int upper_threshold = (faster_motor) ? _r_speed : _l_speed;
    int lower_threshold = (faster_motor) ? _l_speed : _r_speed;

    for (int i = 0; i <= upper_threshold; i++)
    {
      if (i <= lower_threshold)
      {
        if (faster_motor) // right speed is faster
        {
          analogWrite(A_ENABLE, i + 7);
        } else {          // left speed is faster
          analogWrite(B_ENABLE, i);
        }
      }
      if (faster_motor) 
      {
        analogWrite(B_ENABLE, i);
      } else {
        analogWrite(A_ENABLE, i + 7);
      }
      delay(8);
    }
  }
}

/*   CUT POWER SUPPLY TO BOTH MOTORS  */
void stop_motors() 
{ 
  for (int i = 130; i >= 0; i--)
  {
    analogWrite(A_ENABLE, i);
    analogWrite(B_ENABLE, i);
  }
}

void turn_90(bool _r)
{
  set_dir(_r, (1 - _r));
  int l_speed = _r ? 153: 156;
  int r_speed = _r ? 156: 153;
  set_speed(l_speed, r_speed);
  delay(100); 
}

void forward()
{
  set_dir(1, 1);
  set_speed(155, 155);
}

void backward()
{
  set_dir(0, 0);
  set_speed(158, 158);
}

/* PHOTOTRANSISTOR ---------------------------------------------------------------------------------------  */

float get_photo_data(int _pin)
{
  float photo_data = 0.0;
  for (int i = 0; i < 5; i++)
  {
    photo_data += analogRead(_pin);
  }
  return (photo_data / 10.0);
}

/* TEMP SENSOR ---------------------------------------------------------------------------------------  */

float get_temp()
{
  sensors.requestTemperatures();
  return sensors.getTempCByIndex(0);
}

/* BUMP SENSOR ---------------------------------------------------------------------------------------  */

bool get_bump_signal(int _pin)
{
    return digitalRead(_pin);  
}

/* ULTRASONIC SENSOR & SERVO ---------------------------------------------------------------------------------------  */

// supposed range: 2 - 400 cm
float watch()
{
  long echo_distance = 0.0;
  long temp_echo_distance;
  for (int i = 0; i < 10; i++)
  {
    digitalWrite(Trig_PIN,LOW);
    delayMicroseconds(5);                                                                              
    digitalWrite(Trig_PIN,HIGH);
    delayMicroseconds(15);
    digitalWrite(Trig_PIN,LOW);
    temp_echo_distance = pulseIn(Echo_PIN,HIGH);
    echo_distance += ( temp_echo_distance * 0.01657 );   //how far away is the object in cm
  }
  return (float)(echo_distance / 10.0);
}

float get_echo_boolean_at_angle(int _angle)
{
  head.write(_angle);
  delay(50);
  float echo_d = watch();
  return echo_d;
}

void take_action(int _a)
{
  switch (_a)
  {
    case 0: // go forward 
      forward();
      break;
    case 1: // backward 
      backward();
      break;  
    case 2: // go left 
      turn_90(0);
      forward();
      break;
    case 3: // go right 
      turn_90(1);
      forward();
      break;
    case 4: // do nothing
      break; 
  }  
}

void loop() {
  
  // put your main code here, to run repeatedly:
  float l_photo = get_photo_data(l_PHOTO_PIN);
  float r_photo = get_photo_data(r_PHOTO_PIN);
  
  stop_motors();
  
  float temp = get_temp();
  float watch35 = get_echo_boolean_at_angle(35);
  float watch135 = get_echo_boolean_at_angle(135);
  float watch90 = get_echo_boolean_at_angle(90);
  float l_bump = (float) get_bump_signal(l_SWITCH_PIN);
  float r_bump = (float) get_bump_signal(r_SWITCH_PIN);
  
  float data[] = {l_photo, r_photo, temp, watch35, watch135, watch90, l_bump, r_bump};
  String data_string = "";
  
  for (int i = 0; i < 8; i++)
  {
    data_string += String(data[i]);
    data_string += " ";
  }
  data_string += "\n";
  
  Serial.println(data_string);
  
  while (!Serial.available())
  {
    ;  
  }
  
  String a_bytes = Serial.readStringUntil('\n');
  int a = a_bytes.toInt();
  take_action(a);

  
}
