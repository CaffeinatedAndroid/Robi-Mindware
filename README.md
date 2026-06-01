![Alt text](RepoImage.png)
# Robot Mindware
Robot control software and board driver for a custom version of the Deagnostini Robi Robot.
The main software is written in python, it sends ID and position commands to a C++ driver on the ESP32 C3 mini.
The ESP32 then processes commands and sends them to the servo bus, daisy chaining is working.

### Example Command:
```
ID:ANGLE
12:130
15:30
```
![Alt text](Robi Mindware/Documentation/Diagrams/Export/Robi Servo Layout.png)



I am yet to implement the additional serial ports, ideally I would like to work with 5, however that is unlikely. SO the alternative 
will be to use 2 or 3 ports and alternate the less critical ones. That way parts of the robot dont have communications interupted as this could be bad, ie: the legs.

## Hardware:
- Robi servo test board (I will link to an alternative that emulates the same functions in future. It is for setting the ID and testing the motor)
- ESP32 C3 Mini
- 3.3v-5v Bi-directional logic converter (C3 mini outputs 3.3v logic while the servo requires 5v logic, bi-directional opens up the possibility of adding servo feedback)
- 5v Power supply
- OrangePi Zero 2w or Raspi Zero 2w (More ram the better, Recommended minimum is 4GB)
- REMAINING PARTS TBC

### Picking The Robot:
- Robi 1 or 2 
When selecting a Robi, pick a damaged one or a hand assembled one. There is no point in ruining a good unit when you will need to dissasemble it anyway.
Many Robis have dirty contacts, dry servos, damaged or loose servo cables and maybe sometimes a servo that "Forgot" its settings.
By picking a Robi needing some TLC, you will learn far more and not feel guilt for possibly destroying a functional unit.

## Libraries:
- Panda3D
- Numpy
- OpenCV
- Matplotlib
- Tkinter
- Pandas

## Acknowledgments
The following blog from Japan has proven invaluable, The person behind it deserves lots of credit as they have learned a lot of the harder lessons.
Give them a visit and a thanks if you find it useful.

The site is in Japanese so use google translate if needed. They also have a youtube channel.
- https://www.mcc.mbsrv.net/robox/index.html
- https://www.youtube.com/@craftoyaji

