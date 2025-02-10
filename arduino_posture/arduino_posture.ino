// Library needed to use the LCD screen
#include <LiquidCrystal.h>

// Pins for the LCD screen
const int rs = 12, en = 11, d4 = 5, d5 = 4, d6 = 3, d7 = 2;
// LCD object instantiation
LiquidCrystal lcd(rs, en, d4, d5, d6, d7);

// Pins for the RGB LED
int redPin= 6;
int greenPin = 9;
int bluePin = 10;

// Setup function, executed only once when the Arduino is powered on
void setup() {
  // Set the pins as output
  pinMode(redPin, OUTPUT);
  pinMode(greenPin, OUTPUT);
  pinMode(bluePin, OUTPUT);
  // Set the LCD screen size
  lcd.begin(16, 2);
  // Start the serial communication
  Serial.begin(9600);
}

// Function to set the color of the RGB LED
// The function receives the red, green and blue values
// The values are between 0 and 255
// 0 means no color, 255 means full color
// The function uses the analogWrite function to set the color
// The analogWrite function is used to set the PWM signal
// The PWM signal is used to control the brightness of the LED
void setColor(int redValue, int greenValue, int blueValue) {
  analogWrite(redPin, redValue);
  analogWrite(greenPin, greenValue);
  analogWrite(bluePin, blueValue);
}

// Loop function, executed repeatedly
void loop() {
  // Check if there is data available in the serial port
  if (Serial.available()) {
    // Read the data from the serial port
      char serialListener = Serial.read();
      // If the data is 'R', it means that the posture correction algorithm detected a bad posture
      // In this case, the RGB LED will turn red and the LCD screen will display "Warning!"
      if (serialListener == 'R') {
        setColor(255, 0, 0);
        lcd.clear();
        lcd.print("Warning!");
      }
      // If the data is 'G', it means that the posture correction algorithm detected a good posture
      // In this case, the RGB LED will turn green and the LCD screen will display "Ok!"
      else if (serialListener == 'G') {
        setColor(0, 255, 0);
        lcd.clear();
        lcd.print("Ok!");
      }
      // If the data is 'Y', it means that the posture correction algorithm detetcted that the user is not aligned
      // In this case, the RGB LED will turn orange and the LCD screen will display "Not Aligned"
      else if (serialListener == 'Y') {
        setColor(255, 80, 0);
        lcd.clear();
        lcd.print("Not Aligned");
      }
      // If the data is 'V', it means that the posture correction algorithm is computing the posture
      // In this case, the RGB LED will turn purple and the LCD screen will display "Computing..."
      else if (serialListener == 'V') {
        setColor(128, 0, 128);
        lcd.clear();
        lcd.print("Computing...");
      }
      // If the data is 'O', it means that the posture correction algorithm has been stopped
      // In this case, the RGB LED will turn off and the LCD screen will be cleared
      else if (serialListener == 'O') {
        setColor(0, 0, 0);
        lcd.clear();
      }
  }
}