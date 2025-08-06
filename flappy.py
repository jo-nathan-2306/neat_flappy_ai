import neat.config
import pygame
import neat
import time
import os
base_path = os.path.dirname(__file__)
def asset(filename):
    return os.path.join(base_path, "imgs", filename)

import random
pygame.font.init()
winw=500
winh=800
bimage = [
    pygame.transform.scale2x(pygame.image.load(asset("bird1.png"))),
    pygame.transform.scale2x(pygame.image.load(asset("bird2.png"))),
    pygame.transform.scale2x(pygame.image.load(asset("bird3.png")))
]

pipeimage = pygame.transform.scale2x(pygame.image.load(asset("pipe.png")))
baseimage = pygame.transform.scale2x(pygame.image.load(asset("base.png")))
bgimage = pygame.transform.scale2x(pygame.image.load(asset("bg.png")))
statfont = pygame.font.SysFont("comicsans",50)

GEN=0

class Bird:
    imgs=bimage
    max_rotation=25
    rotvel=20
    animationt=5
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.tilt=0
        self.tick_count=0
        self.vel=0
        self.height=self.y
        self.imgc=0
        self.img=self.imgs[0]
    def jump(self):
        self.vel=-10.5
        self.tick_count=0
        self.height=self.y
    def move(self):
        self.tick_count+=1
        d=self.vel*self.tick_count+1.5*self.tick_count**2
        if d>=16:
            d=16
        if d<0:
            d-=2
        self.y=self.y+d
        if d<0 or self.y<self.height+50:
            if self.tilt<self.max_rotation:
                self.tilt=self.max_rotation
        else:
            if self.tilt>-90:
                self.tilt-=self.rotvel
    def draw(self,win):
        self.imgc+=1
        if self.imgc<self.animationt:
            self.img=self.imgs[0]
        elif self.imgc<self.animationt*2:
            self.img=self.imgs[1]
        elif self.imgc<self.animationt*3:
            self.img=self.imgs[2]
        elif self.imgc<self.animationt*4:
            self.img=self.imgs[1]
        elif self.imgc==self.animationt*4+1:
            self.img=self.imgs[0]
            self.imgc=0
        if self.tilt<=-80:
            self.img=self.imgs[1]
            self.imgc=self.animationt*2
        rimage=pygame.transform.rotate(self.img,self.tilt)
        rect=rimage.get_rect(center=self.img.get_rect(topleft=(self.x,self.y)).center)
        win.blit(rimage,rect.topleft)
    def get_mask(self):
        return pygame.mask.from_surface(self.img)
class Pipe:
    GAP=200
    VEL=5
    def __init__(self,x):
        self.x=x
        self.height=0
        self.top=0
        self.bottom=0
        self.toppipe=pygame.transform.flip(pipeimage,False,True)
        self.bottompipe=pipeimage
        self.passed=False
        self.seth()
    def seth(self):
        self.height=random.randrange(50,450)
        self.top=self.height-self.toppipe.get_height()
        self.bottom=self.height+self.GAP
    def move(self):
        self.x-=self.VEL
    def draw(self,win):
        win.blit(self.toppipe,(self.x,self.top))
        win.blit(self.bottompipe,(self.x,self.bottom))
    def collide(self,bird):
        birdmask=bird.get_mask()   
        topmask=pygame.mask.from_surface(self.toppipe)
        bottommask=pygame.mask.from_surface(self.bottompipe)
        topoffset=(self.x-bird.x,self.top-round(bird.y))
        bottomoffset=(self.x-bird.x,self.bottom-round(bird.y))
        bpoint=birdmask.overlap(bottommask,bottomoffset)
        tpoint=birdmask.overlap(topmask,topoffset)
        if tpoint or bpoint:
            return True
        return False
class Base:
    VEL=5
    width=baseimage.get_width()
    IMG=baseimage
    def __init__(self,y):
        self.y=y
        self.x1=0
        self.x2=self.width
    def move(self):
        self.x1-=self.VEL
        self.x2-=self.VEL
        if self.x1+self.width<0:
            self.x1=self.x2+self.width
        if self.x2+self.width<0:
            self.x2=self.x1+self.width
    def draw(self,win):
        win.blit(self.IMG,(self.x1,self.y))
        win.blit(self.IMG,(self.x2,self.y))
def draw_window(win,birds,pipes,base,score,gen):
    win.blit(bgimage,(0,0))
    txt=statfont.render("Score "+ str(score),1,(255,255,255))
    win.blit(txt,(winw-10-txt.get_width(),10))
    txt=statfont.render("GEN: "+ str(gen),1,(255,255,255))
    win.blit(txt,(10,10))
    for pipe in pipes:
        pipe.draw(win)
    base.draw(win)
    for bird in birds:
        bird.draw(win)
    pygame.display.update()
def main(genomes,config):
    birds=[]
    nets=[]
    ge=[]
    global GEN
    GEN+=1
    for _,g in genomes:
        net=neat.nn.FeedForwardNetwork.create(g,config)
        nets.append(net)
        birds.append(Bird(230,350))
        g.fitness=0
        ge.append(g)
    base=Base(730)
    pipes=[Pipe(700)]
    run=True
    score=0
    clock=pygame.time.Clock()
    win=pygame.display.set_mode((winw,winh))
    pipein=0
    while run:
        clock.tick(30)
        rem=[]
        add_pipe=False
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                run=False
                pygame.quit()
                quit()
        pipeind=0
        if len(birds)>0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].toppipe.get_width():
                pipeind = 1

        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.1
            output=nets[x].activate((bird.y,abs(bird.y-pipes[pipeind].height),abs(bird.y-pipes[pipeind].bottom)))
            if output[0]>0.5:
                bird.jump()
        for pipe in pipes:
            for x,bird in enumerate(birds):
                if pipe.collide(bird):
                    ge[x].fitness-=1
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)
                if not pipe.passed and pipe.x<bird.x:
                    pipe.passed=True
                    add_pipe=True
            if pipe.x+pipe.toppipe.get_width()<0:
                rem.append(pipe)
            pipe.move()
        if add_pipe:
            score+=1
            for g in ge:
                g.fitness+=5
            pipes.append(Pipe(600))
        for r in rem:
            pipes.remove(r)
        for x,bird in enumerate(birds):
            if bird.y+bird.img.get_height()>=730 or bird.y<0:
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)
        if len(birds) == 0:
            return

        #bird.move() 
        base.move()
        draw_window(win,birds,pipes,base,score,GEN)

def run():
    config=neat.Config(neat.DefaultGenome,neat.DefaultReproduction,neat.DefaultSpeciesSet,neat.DefaultStagnation,config_path)
    p=neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stat=neat.StatisticsReporter()
    p.add_reporter(stat)
    winner=p.run(main,50)
if __name__=="__main__":
    local_dir=os.path.dirname(__file__)
    config_path=os.path.join(local_dir,"config.txt")
    run() 
            


