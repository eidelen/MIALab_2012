# This region grow implementation was taken from mryigit on
# https://gist.github.com/1453329 and adapted to work with numpy 
# arrays instead of PIL images. 
# Adrian Schneider

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
        

import numpy as np

def regiongrow(img,epsilon,start_point):
    
    image = img.astype(np.float)     
    
    Q = Queue()
    s = []
    
    y = start_point[0]
    x = start_point[1]
    
    
    Q.enque((y,x))

    
    while not Q.isEmpty():

        t = Q.deque()
        y = t[0]
        x = t[1]

        height, width = image.shape
        
        if x < (width-1) and abs(  image[y , x + 1] - image[y , x]  ) <= epsilon :

            if not Q.isInside( (y , x + 1) ) and not (y , x + 1) in s:
                Q.enque( (y , x + 1) )

                
        if x > 0 and abs(  image[y , x - 1] - image[y , x]  ) <= epsilon:

                if not Q.isInside( (y , x - 1) ) and not (y , x - 1) in s:
                    Q.enque( (y , x - 1) )

                     
        if y < (height - 1) and \
           abs(  image[y + 1 , x]- image[y , x]  ) <= epsilon:

            if not Q.isInside( (y + 1, x) ) and not (y + 1 , x) in s:
                Q.enque( (y + 1 , x) )

                    
        if y > 0 and \
           abs(  image[y - 1 , x] - image[y , x]  ) <= epsilon:

            if not Q.isInside( (y - 1 , x) ) and not (y - 1 , x) in s:
                Q.enque( (y - 1 , x) )


        if t not in s:
            s.append( t )


    region = np.zeros( (height, width) )
    
    for i in range ( width ):
        for j in range ( height ):
            region[j , i] = 0 

    print np.size(s)
    
    for i in s:
        region[i[0], i[1]] = 150
        
    return region

