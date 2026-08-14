#!/usr/bin/python
# -*- coding: UTF-8 -*-
#import chardet
import os
import sys 
import time
import logging
import spidev as SPI
sys.path.append("..")
from lib import LCD_1inch28
from PIL import Image,ImageDraw,ImageFont, ImageChops

# Raspberry Pi pin configuration:
RST = 27
DC = 25
BL = 18
bus = 0 
device = 0 
freq = 99999999

#EYE CONFIG
ring_COLOR = "PURPLE"

pupil_COLOR = "WHITE"

offsetx = 0
offsety = -1

offsetxL = -10
offsetyL = -2

leftArc = 10,10,228,228
rightArc = 5,5,218,218

def BLINK():
    
        time.sleep(5)
        
 #############################################       
        
        disp = LCD_1inch28.LCD_1inch28(spi=SPI.SpiDev(bus, 0),spi_freq=freq,rst=RST,dc=DC,bl=BL)
        disp.Init()
        maskcolour = Image.new("RGBA", (disp.width, disp.height), pupil_COLOR)
        image1 = Image.open('../pic/openMid.jpg')
        mask = Image.open('../pic/openMid.jpg').convert('L')
        image2 = ImageChops.offset(image1,offsetxL,offsetyL)
        maskfix = ImageChops.offset(mask,offsetxL,offsetyL)
        im = Image.composite(maskcolour, image2, maskfix)
        imagefinal = im

        draw = ImageDraw.Draw(imagefinal)
        
        draw.arc((rightArc),0, 360, fill =(ring_COLOR), width =20)

        im_r=imagefinal.rotate(90)
        disp.ShowImage(im_r)
        
        disp.module_exit()   
        
#############################################        
        
        disp = LCD_1inch28.LCD_1inch28(spi=SPI.SpiDev(bus, 1),spi_freq=freq,rst=RST,dc=DC,bl=BL)
        disp.Init()
        maskcolour = Image.new("RGBA", (disp.width, disp.height), pupil_COLOR)
        image1 = Image.open('../pic/openMid.jpg')
        mask = Image.open('../pic/openMid.jpg').convert('L')
        image2 = ImageChops.offset(image1,offsetx,offsety)
        maskfix = ImageChops.offset(mask,offsetx,offsety)
        im = Image.composite(maskcolour, image2, maskfix)
        imagefinal = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        draw = ImageDraw.Draw(imagefinal)
        
        draw.arc((leftArc),0, 360, fill =(ring_COLOR), width =20)

        im_r=imagefinal.rotate(-90)
        disp.ShowImage(im_r)

        disp.module_exit()          
        
#############################################        
                
        disp = LCD_1inch28.LCD_1inch28(spi=SPI.SpiDev(bus, 0),spi_freq=freq,rst=RST,dc=DC,bl=BL)
        disp.Init()
        maskcolour = Image.new("RGBA", (disp.width, disp.height), pupil_COLOR)
        image1 = Image.open('../pic/half.jpg')
        mask = Image.open('../pic/half.jpg').convert('L')
        image2 = ImageChops.offset(image1,offsetxL,offsetyL)
        maskfix = ImageChops.offset(mask,offsetxL,offsetyL)
        im = Image.composite(maskcolour, image2, maskfix)
        imagefinal = im

        draw = ImageDraw.Draw(imagefinal)
        
        draw.arc((rightArc),0, 360, fill =(ring_COLOR), width =20)

        im_r=imagefinal.rotate(90)
        disp.ShowImage(im_r)
        
        disp.module_exit()   
        
#############################################        
        
        disp = LCD_1inch28.LCD_1inch28(spi=SPI.SpiDev(bus, 1),spi_freq=freq,rst=RST,dc=DC,bl=BL)
        disp.Init()
        maskcolour = Image.new("RGBA", (disp.width, disp.height), pupil_COLOR)
        image1 = Image.open('../pic/half.jpg')
        mask = Image.open('../pic/half.jpg').convert('L')
        image2 = ImageChops.offset(image1,offsetx,offsety)
        maskfix = ImageChops.offset(mask,offsetx,offsety)
        im = Image.composite(maskcolour, image2, maskfix)
        imagefinal = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        draw = ImageDraw.Draw(imagefinal)
        
        draw.arc((leftArc),0, 360, fill =(ring_COLOR), width =20)

        im_r=imagefinal.rotate(-90)
        disp.ShowImage(im_r)

        disp.module_exit()        
        
#############################################        
        
        disp = LCD_1inch28.LCD_1inch28(spi=SPI.SpiDev(bus, 0),spi_freq=freq,rst=RST,dc=DC,bl=BL)
        disp.Init()
        maskcolour = Image.new("RGBA", (disp.width, disp.height), pupil_COLOR)
        image1 = Image.open('../pic/closed.jpg')
        mask = Image.open('../pic/closed.jpg').convert('L')
        image2 = ImageChops.offset(image1,offsetxL,offsetyL)
        maskfix = ImageChops.offset(mask,offsetxL,offsetyL)
        im = Image.composite(maskcolour, image2, maskfix)
        imagefinal = im

        draw = ImageDraw.Draw(imagefinal)
        
        draw.arc((rightArc),0, 360, fill =(ring_COLOR), width =20)

        im_r=imagefinal.rotate(90)
        disp.ShowImage(im_r)
        
        disp.module_exit()   
        
#############################################        
        
        disp = LCD_1inch28.LCD_1inch28(spi=SPI.SpiDev(bus, 1),spi_freq=freq,rst=RST,dc=DC,bl=BL)
        disp.Init()
        maskcolour = Image.new("RGBA", (disp.width, disp.height), pupil_COLOR)
        image1 = Image.open('../pic/closed.jpg')
        mask = Image.open('../pic/closed.jpg').convert('L')
        image2 = ImageChops.offset(image1,offsetx,offsety)
        maskfix = ImageChops.offset(mask,offsetx,offsety)
        im = Image.composite(maskcolour, image2, maskfix)
        imagefinal = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        draw = ImageDraw.Draw(imagefinal)
        
        draw.arc((leftArc),0, 360, fill =(ring_COLOR), width =20)

        im_r=imagefinal.rotate(-90)
        disp.ShowImage(im_r)
        
        disp.module_exit() 
        
#############################################        
        
        disp = LCD_1inch28.LCD_1inch28(spi=SPI.SpiDev(bus, 0),spi_freq=freq,rst=RST,dc=DC,bl=BL)
        disp.Init()
        maskcolour = Image.new("RGBA", (disp.width, disp.height), pupil_COLOR)
        image1 = Image.open('../pic/half.jpg')
        mask = Image.open('../pic/half.jpg').convert('L')
        image2 = ImageChops.offset(image1,offsetxL,offsetyL)
        maskfix = ImageChops.offset(mask,offsetxL,offsetyL)
        im = Image.composite(maskcolour, image2, maskfix)
        imagefinal = im

        draw = ImageDraw.Draw(imagefinal)
        
        draw.arc((rightArc),0, 360, fill =(ring_COLOR), width =20)

        im_r=imagefinal.rotate(90)
        disp.ShowImage(im_r)
        
        disp.module_exit()   
        
#############################################        
        
        disp = LCD_1inch28.LCD_1inch28(spi=SPI.SpiDev(bus, 1),spi_freq=freq,rst=RST,dc=DC,bl=BL)
        disp.Init()
        maskcolour = Image.new("RGBA", (disp.width, disp.height), pupil_COLOR)
        image1 = Image.open('../pic/half.jpg')
        mask = Image.open('../pic/half.jpg').convert('L')
        image2 = ImageChops.offset(image1,offsetx,offsety)
        maskfix = ImageChops.offset(mask,offsetx,offsety)
        im = Image.composite(maskcolour, image2, maskfix)
        imagefinal = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        draw = ImageDraw.Draw(imagefinal)
        
        draw.arc((leftArc),0, 360, fill =(ring_COLOR), width =20)

        im_r=imagefinal.rotate(-90)
        disp.ShowImage(im_r)
        
        disp.module_exit()        
        
#############################################        

        disp = LCD_1inch28.LCD_1inch28(spi=SPI.SpiDev(bus, 1),spi_freq=freq,rst=RST,dc=DC,bl=BL)
        disp.Init()
        maskcolour = Image.new("RGBA", (disp.width, disp.height), pupil_COLOR)
        image1 = Image.open('../pic/open.jpg')
        mask = Image.open('../pic/open.jpg').convert('L')
        image2 = ImageChops.offset(image1,offsetx,offsety)
        maskfix = ImageChops.offset(mask,offsetx,offsety)
        im = Image.composite(maskcolour, image2, maskfix)
        imagefinal = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        draw = ImageDraw.Draw(imagefinal)
        
        draw.arc((leftArc),0, 360, fill =(ring_COLOR), width =20)

        im_r=imagefinal.rotate(-90)
        disp.ShowImage(im_r)
        disp.module_exit()       
         
#############################################         
            
        disp = LCD_1inch28.LCD_1inch28(spi=SPI.SpiDev(bus, 0),spi_freq=freq,rst=RST,dc=DC,bl=BL)
        disp.Init()
        maskcolour = Image.new("RGBA", (disp.width, disp.height), pupil_COLOR)
        image1 = Image.open('../pic/open.jpg')
        mask = Image.open('../pic/open.jpg').convert('L')
        image2 = ImageChops.offset(image1,offsetxL,offsetyL)
        maskfix = ImageChops.offset(mask,offsetxL,offsetyL)
        im = Image.composite(maskcolour, image2, maskfix)
        imagefinal = im

        draw = ImageDraw.Draw(imagefinal)
        
        draw.arc((rightArc),0, 360, fill =(ring_COLOR), width =20)

        im_r=imagefinal.rotate(90)
        disp.ShowImage(im_r)
        
        disp.module_exit()

def HAPPY_BLINK():
    
        time.sleep(5)
        
#############################################        
        
        disp = LCD_1inch28.LCD_1inch28(spi=SPI.SpiDev(bus, 0),spi_freq=freq,rst=RST,dc=DC,bl=BL)
        disp.Init()
        maskcolour = Image.new("RGBA", (disp.width, disp.height), pupil_COLOR)
        image1 = Image.open('../pic/happy1.jpg')
        mask = Image.open('../pic/happy1.jpg').convert('L')
        image2 = ImageChops.offset(image1,offsetxL,offsetyL)
        maskfix = ImageChops.offset(mask,offsetxL,offsetyL)
        im = Image.composite(maskcolour, image2, maskfix)
        imagefinal = im

        draw = ImageDraw.Draw(imagefinal)
        
        draw.arc((rightArc),0, 360, fill =(ring_COLOR), width =20)

        im_r=imagefinal.rotate(90)
        disp.ShowImage(im_r)
        
        disp.module_exit()   
        
        
#############################################
        
        disp = LCD_1inch28.LCD_1inch28(spi=SPI.SpiDev(bus, 1),spi_freq=freq,rst=RST,dc=DC,bl=BL)
        disp.Init()
        maskcolour = Image.new("RGBA", (disp.width, disp.height), pupil_COLOR)
        image1 = Image.open('../pic/happy1.jpg')
        mask = Image.open('../pic/happy1.jpg').convert('L')
        image2 = ImageChops.offset(image1,offsetx,offsety)
        maskfix = ImageChops.offset(mask,offsetx,offsety)
        im = Image.composite(maskcolour, image2, maskfix)
        imagefinal = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        draw = ImageDraw.Draw(imagefinal)
        
        draw.arc((leftArc),0, 360, fill =(ring_COLOR), width =20)

        im_r=imagefinal.rotate(-90)
        disp.ShowImage(im_r)
        #time.sleep(0.5)

        disp.module_exit()          
        
 #############################################       
                
        disp = LCD_1inch28.LCD_1inch28(spi=SPI.SpiDev(bus, 0),spi_freq=freq,rst=RST,dc=DC,bl=BL)
        disp.Init()
        maskcolour = Image.new("RGBA", (disp.width, disp.height), pupil_COLOR)
        image1 = Image.open('../pic/happy2.jpg')
        mask = Image.open('../pic/happy2.jpg').convert('L')
        image2 = ImageChops.offset(image1,offsetxL,offsetyL)
        maskfix = ImageChops.offset(mask,offsetxL,offsetyL)
        im = Image.composite(maskcolour, image2, maskfix)
        imagefinal = im

        draw = ImageDraw.Draw(imagefinal)
        
        draw.arc((rightArc),0, 360, fill =(ring_COLOR), width =20)

        im_r=imagefinal.rotate(90)
        disp.ShowImage(im_r)
        

        disp.module_exit()   
 #############################################       
        disp = LCD_1inch28.LCD_1inch28(spi=SPI.SpiDev(bus, 1),spi_freq=freq,rst=RST,dc=DC,bl=BL)
        disp.Init()
        maskcolour = Image.new("RGBA", (disp.width, disp.height), pupil_COLOR)
        image1 = Image.open('../pic/happy2.jpg')
        mask = Image.open('../pic/happy2.jpg').convert('L')
        image2 = ImageChops.offset(image1,offsetx,offsety)
        maskfix = ImageChops.offset(mask,offsetx,offsety)
        im = Image.composite(maskcolour, image2, maskfix)
        imagefinal = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        draw = ImageDraw.Draw(imagefinal)
        
        draw.arc((leftArc),0, 360, fill =(ring_COLOR), width =20)

        im_r=imagefinal.rotate(-90)
        disp.ShowImage(im_r)

        disp.module_exit()    
            
        time.sleep(1)
        
 ############################################# 
       
        disp = LCD_1inch28.LCD_1inch28(spi=SPI.SpiDev(bus, 0),spi_freq=freq,rst=RST,dc=DC,bl=BL)
        disp.Init()
        maskcolour = Image.new("RGBA", (disp.width, disp.height), pupil_COLOR)
        image1 = Image.open('../pic/happy1.jpg')
        mask = Image.open('../pic/happy1.jpg').convert('L')
        image2 = ImageChops.offset(image1,offsetxL,offsetyL)
        maskfix = ImageChops.offset(mask,offsetxL,offsetyL)
        im = Image.composite(maskcolour, image2, maskfix)
        imagefinal = im

        draw = ImageDraw.Draw(imagefinal)
        
        draw.arc((rightArc),0, 360, fill =(ring_COLOR), width =20)

        im_r=imagefinal.rotate(90)
        disp.ShowImage(im_r)
        

        disp.module_exit()   
 #############################################       
        disp = LCD_1inch28.LCD_1inch28(spi=SPI.SpiDev(bus, 1),spi_freq=freq,rst=RST,dc=DC,bl=BL)
        disp.Init()
        maskcolour = Image.new("RGBA", (disp.width, disp.height), pupil_COLOR)
        image1 = Image.open('../pic/happy1.jpg')
        mask = Image.open('../pic/happy1.jpg').convert('L')
        image2 = ImageChops.offset(image1,offsetx,offsety)
        maskfix = ImageChops.offset(mask,offsetx,offsety)
        im = Image.composite(maskcolour, image2, maskfix)
        imagefinal = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        draw = ImageDraw.Draw(imagefinal)
        
        draw.arc((leftArc),0, 360, fill =(ring_COLOR), width =20)

        im_r=imagefinal.rotate(-90)
        disp.ShowImage(im_r)

        disp.module_exit()          
        
#############################################        
                
        disp = LCD_1inch28.LCD_1inch28(spi=SPI.SpiDev(bus, 0),spi_freq=freq,rst=RST,dc=DC,bl=BL)
        disp.Init()
        maskcolour = Image.new("RGBA", (disp.width, disp.height), pupil_COLOR)
        image1 = Image.open('../pic/happy2.jpg')
        mask = Image.open('../pic/happy2.jpg').convert('L')
        image2 = ImageChops.offset(image1,offsetxL,offsetyL)
        maskfix = ImageChops.offset(mask,offsetxL,offsetyL)
        im = Image.composite(maskcolour, image2, maskfix)
        imagefinal = im

        draw = ImageDraw.Draw(imagefinal)
        
        draw.arc((rightArc),0, 360, fill =(ring_COLOR), width =20)

        im_r=imagefinal.rotate(90)
        disp.ShowImage(im_r)
        
        disp.module_exit()   
        
 #############################################   
     
        disp = LCD_1inch28.LCD_1inch28(spi=SPI.SpiDev(bus, 1),spi_freq=freq,rst=RST,dc=DC,bl=BL)
        disp.Init()
        maskcolour = Image.new("RGBA", (disp.width, disp.height), pupil_COLOR)
        image1 = Image.open('../pic/happy2.jpg')
        mask = Image.open('../pic/happy2.jpg').convert('L')
        image2 = ImageChops.offset(image1,offsetx,offsety)
        maskfix = ImageChops.offset(mask,offsetx,offsety)
        im = Image.composite(maskcolour, image2, maskfix)
        imagefinal = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        draw = ImageDraw.Draw(imagefinal)
        
        draw.arc((leftArc),0, 360, fill =(ring_COLOR), width =20)

        im_r=imagefinal.rotate(-90)
        disp.ShowImage(im_r)
        
        disp.module_exit()   

try:

    count = 0
    while (count == 0):
        count = count 
        
        BLINK()
        HAPPY_BLINK()

 #############################################
        #TESTING
    
  
 #############################################
        
        disp = LCD_1inch28.LCD_1inch28(spi=SPI.SpiDev(bus, 0),spi_freq=freq,rst=RST,dc=DC,bl=BL)
        disp.Init()
        maskcolour = Image.new("RGBA", (disp.width, disp.height), pupil_COLOR)

        image1 = Image.open('../pic/open.jpg')
        mask = Image.open('../pic/open.jpg').convert('L')
        image2 = ImageChops.offset(image1,offsetxL,offsetyL)
        maskfix = ImageChops.offset(mask,offsetxL,offsetyL)
        im = Image.composite(maskcolour, image2, maskfix)
        imagefinal = im

        draw = ImageDraw.Draw(imagefinal)
        
        draw.arc((rightArc),0, 360, fill =(ring_COLOR), width =20)

        im_r=imagefinal.rotate(90)
        disp.ShowImage(im_r)
        

        disp.module_exit()   
        
#############################################        
        
        disp = LCD_1inch28.LCD_1inch28(spi=SPI.SpiDev(bus, 1),spi_freq=freq,rst=RST,dc=DC,bl=BL)
        disp.Init()
        maskcolour = Image.new("RGBA", (disp.width, disp.height), pupil_COLOR)

        image1 = Image.open('../pic/open.jpg')
        mask = Image.open('../pic/open.jpg').convert('L')
        image2 = ImageChops.offset(image1,offsetx,offsety)
        maskfix = ImageChops.offset(mask,offsetx,offsety)
        im = Image.composite(maskcolour, image2, maskfix)
        imagefinal = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        draw = ImageDraw.Draw(imagefinal)
        
        draw.arc((leftArc),0, 360, fill =(ring_COLOR), width =20)

        im_r=imagefinal.rotate(-90)
        disp.ShowImage(im_r)

        disp.module_exit()        
        
#############################################

        disp = LCD_1inch28.LCD_1inch28(spi=SPI.SpiDev(bus, 1),spi_freq=freq,rst=RST,dc=DC,bl=BL)
        disp.Init()
        maskcolour = Image.new("RGBA", (disp.width, disp.height), pupil_COLOR)

        image1 = Image.open('../pic/open.jpg')
        mask = Image.open('../pic/open.jpg').convert('L')
        image2 = ImageChops.offset(image1,offsetx,offsety)
        maskfix = ImageChops.offset(mask,offsetx,offsety)
        im = Image.composite(maskcolour, image2, maskfix)
        imagefinal = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        draw = ImageDraw.Draw(imagefinal)
        
        draw.arc((leftArc),0, 360, fill =(ring_COLOR), width =20)

        im_r=imagefinal.rotate(-90)
        disp.ShowImage(im_r)
        disp.module_exit()       
        
 #############################################       
            
        disp = LCD_1inch28.LCD_1inch28(spi=SPI.SpiDev(bus, 0),spi_freq=freq,rst=RST,dc=DC,bl=BL)
        disp.Init()
        maskcolour = Image.new("RGBA", (disp.width, disp.height), pupil_COLOR)
        image1 = Image.open('../pic/open.jpg')
        mask = Image.open('../pic/open.jpg').convert('L')
        image2 = ImageChops.offset(image1,offsetxL,offsetyL)
        maskfix = ImageChops.offset(mask,offsetxL,offsetyL)
        im = Image.composite(maskcolour, image2, maskfix)
        imagefinal = im

        draw = ImageDraw.Draw(imagefinal)
        
        draw.arc((rightArc),0, 360, fill =(ring_COLOR), width =20)

        im_r=imagefinal.rotate(90)
        disp.ShowImage(im_r)
        
      
        disp.module_exit()
            
    else:
        null
              
except KeyboardInterrupt:
    disp.module_exit()
    #logging.info("quit:")
    exit()
