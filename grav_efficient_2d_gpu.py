import pygame
from torch import cat, nan_to_num, sum, ones, einsum, sqrt, randint, float64, int64, stack

pygame.init()

G = 6.67430E-11
FPS = 20
SPF = 1E-1/FPS
RADIUS = 3

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
WIDTH, HEIGHT = 1000, 1000
DISH = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('2D Gravity Simulator Efficient CUDA')
FONT = pygame.font.SysFont("Arial" , 18 , bold = True)
clock = pygame.time.Clock()

EPSILON = 1E-1


def newtonian_gravitational_dynamics(ringo, color, counter, M, SPF=1/144, WIDTH=600, HEIGHT=600):
    x = ringo[:, :2]    
    y = x.reshape(x.shape[0], 1, x.shape[1])
    
    R = sqrt(einsum('ijk, ijk->ij', x-y, x-y))  
    M_ = M @ M.T
    R_ = stack([(x - x[i]) / (R[i].reshape(R.shape[0], 1) + EPSILON) for i in counter],dim=0)
    
    a = sum(nan_to_num((1/(M * ones((R.shape[0], R.shape[0])).cuda())) * G * (M_/((R ** 2) + EPSILON))).reshape(R.shape[0], R.shape[0], 1) * R_, axis=1)
    v = ringo[:, 2:] + (a * SPF)
    x = x + (v * SPF)
    
    # Boundry Condition
    #v = v * cat((where(((x[:,0] > WIDTH) * v[:,0]) > 0, -1, 1).reshape(x.shape[0], 1),
    #                    where(((x[:,1] > HEIGHT) * v[:,1]) > 0, -1, 1).reshape(x.shape[0], 1)),1) *\
    #        cat((where(((x[:,0] < 0) * v[:,0]) < 0, -1, 1).reshape(x.shape[0], 1),
    #                    where(((x[:,1] < 0) * v[:,1]) < 0, -1, 1).reshape(x.shape[0], 1)),1)
    
    # Outputs
    ringo = cat((x, v), axis=1)
    arty = cat((x.to(dtype = int64), color), axis=1)
    return arty, ringo


def fps_counter():
    flag = int(clock.get_fps())
    c = "RED" if flag < 5 else "YELLOW" if flag < 10 else "GREEN"
    fps = str(flag)
    fps_t = FONT.render(fps , 1, pygame.Color(c))
    DISH.blit(fps_t,(0,0))


def draw_window(arty, lighting):
    DISH.fill(lighting)
    [pygame.draw.circle(DISH, [cell[2], cell[3], cell[4]], (cell[0], cell[1]), RADIUS) if cell[0]>0 and cell[0]<WIDTH and cell[1]>0 and cell[1]<HEIGHT else '' for cell in arty.detach().cpu().numpy()]
    fps_counter()
    pygame.display.update()


def main():
    run = True
    
    N = 1000
    M = randint(int(3E7), int(4E7), (N, 1)).to(dtype = float64).cuda() 
    M[0] = M[0]*1E10
    
    width = randint(160, 370, (N, 1)).to(dtype = float64).cuda()
    width[0] = WIDTH/2
    height = randint(160, 370, (N, 1)).to(dtype = float64).cuda()
    height[0] = HEIGHT/2
    
    vx = randint(240, 241, (N, 1)).to(dtype = float64).cuda()
    vy = randint(3, 4, (N, 1)).to(dtype = float64).cuda() * 0.
    vx[0] = 0.
    vy[0] = 0.
    
    ringo = cat((width, height, vx, vy), axis=1).cuda()
    color = randint(int(150), int(255), (N, 3)).cuda()
    
    time_step = 0
    lighting = BLACK
    counter = range(N)

    while run:
        clock.tick(FPS)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                
        arty, ringo = newtonian_gravitational_dynamics(ringo, color, counter, M = M, WIDTH = WIDTH, HEIGHT = HEIGHT, SPF = SPF)
        draw_window(arty, lighting)
        time_step += 1
        
    pygame.quit()


if __name__ == '__main__':
    main()