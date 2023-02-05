import pygame
from torch import cat, nan_to_num, where, zeros, sum, ones, einsum, sqrt, randint, float32, int32, stack

G = 6.67430E-11
FPS = 240
SPF = 1/FPS
RADIUS = 3

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
WIDTH, HEIGHT = 900, 900
DISH = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('2D Gravity Simulator Efficient CUDA')

EPSILON = 1E-1


def newtonian_gravitational_dynamics(ringo, color, counter, M, R_, SPF=1/144, WIDTH=600, HEIGHT=600):
    x = ringo[:, :2]    
    y = x.reshape(x.shape[0], 1, x.shape[1])
    
    R = sqrt(einsum('ijk, ijk->ij', x-y, x-y))  
    M_ = M @ M.T
    R_ = stack([(x - x[i]) / (R[i].reshape(R.shape[0], 1) + EPSILON) for i in counter],dim=0)
    
    a = sum(nan_to_num((1/(M * ones((R.shape[0], R.shape[0])).cuda())) * G * (M_/((R ** 2) + EPSILON))).reshape(R.shape[0], R.shape[0], 1) * R_, axis=1)
    v = ringo[:, 2:] + (a * SPF)
    x = x + (v * SPF)
    
    # Boundry Condition
    v = v * cat((where(((x[:,0] > WIDTH) * v[:,0]) > 0, -1, 1).reshape(x.shape[0], 1),
                        where(((x[:,1] > HEIGHT) * v[:,1]) > 0, -1, 1).reshape(x.shape[0], 1)),1) *\
            cat((where(((x[:,0] < 0) * v[:,0]) < 0, -1, 1).reshape(x.shape[0], 1),
                        where(((x[:,1] < 0) * v[:,1]) < 0, -1, 1).reshape(x.shape[0], 1)),1)
    
    # Outputs
    ringo = cat((x, v), axis=1)
    arty = cat((x.to(dtype = int32), color), axis=1)
    return arty, ringo


def draw_window(arty, lighting):
    DISH.fill(lighting)
    [pygame.draw.circle(DISH, [cell[2], cell[3], cell[4]], (cell[0], cell[1]), RADIUS) for cell in arty.detach().cpu().numpy()]
    pygame.display.update()


def main():
    clock = pygame.time.Clock()
    run = True
    N = 2000
    M = randint(int(3E7), int(4E7), (N, 1)).to(dtype = float32).cuda()
    M[0] = M[0]*1E10
    width = randint(380, 381, (N, 1)).to(dtype = float32).cuda()
    width[0] = WIDTH/2
    height = randint(260, 370, (N, 1)).to(dtype = float32).cuda()
    height[0] = HEIGHT/2
    vx = randint(240, 241, (N, 1)).to(dtype = float32).cuda()
    vy = randint(3, 4, (N, 1)).to(dtype = float32).cuda() * 0.
    vx[0] = 0.
    vy[0] = 0.
    ringo = cat((width, height, vx, vy), axis=1).cuda()
    color = randint(int(150), int(255), (N, 3)).cuda()
    time_step = 0
    lighting = BLACK
    counter = range(N)
    R_ = zeros((ringo[:, :2].shape[0], ringo[:, :2].shape[0], ringo[:, :2].shape[1])).cuda()

    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        arty, ringo = newtonian_gravitational_dynamics(ringo, color, counter, M = M, R_ = R_, WIDTH = WIDTH, HEIGHT = HEIGHT, SPF = SPF)
        draw_window(arty, lighting)
        time_step += 1
    pygame.quit()


if __name__ == '__main__':
    main()
