class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items==[]

    def enque(self,item):
        self.items.insert(0,item)

    def deque(self):
        return self.items.pop()

    def qsize(self):
        return len(self.items)
    
    def isInside(self, item):
        return (item in self.items)
        

import Image,os
import numpy as np

def regiongrow(image,epsilon,start_point):
    
    newimage = np.copy(image)
    
    Q = Queue()
    s = []
    
    x = start_point[0]
    y = start_point[1]
    
    Q.enque((x,y))

    
    while not Q.isEmpty():

        t = Q.deque()
        x = t[0]
        y = t[1]
        
        height, width = image.shape
        if x < width-1 and \
           abs(  image[x + 1 , y] - image[x , y]  ) <= epsilon :

            if not Q.isInside( (x + 1 , y) ) and not (x + 1 , y) in s:
                Q.enque( (x + 1 , y) )

                
        if x > 0 and \
           abs(  image[x - 1 , y] - image[x , y]  ) <= epsilon:

            if not Q.isInside( (x - 1 , y) ) and not (x - 1 , y) in s:
                Q.enque( (x - 1 , y) )

                     
        if y < (height - 1) and \
           abs(  image[x , y + 1]- image[x , y]  ) <= epsilon:

            if not Q.isInside( (x, y + 1) ) and not (x , y + 1) in s:
                Q.enque( (x , y + 1) )

                    
        if y > 0 and \
           abs(  image[x , y - 1] - image[x , y]  ) <= epsilon:

            if not Q.isInside( (x , y - 1) ) and not (x , y - 1) in s:
                Q.enque( (x , y - 1) )


        if t not in s:
            s.append( t )

            
    #image.load()
    #putpixel = image.im.putpixel
    
    for i in range ( width ):
        for j in range ( height ):
            newimage[j , i] = 0 

    for i in s:
        newimage[i[1], i[0]] = 150
        
    return newimage

    #output=raw_input("enter save fle name : ")
    #image.thumbnail( (image.size[0] , image.size[1]) , Image.ANTIALIAS )
    #image.save(output + ".JPEG" , "JPEG")
