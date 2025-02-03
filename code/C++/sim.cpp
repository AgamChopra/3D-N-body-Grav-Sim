// main.cpp
#include <SDL2/SDL.h>
#include <SDL2/SDL2_gfxPrimitives.h>
#include <SDL2/SDL_ttf.h>
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// ---------------- Global constants and variables ----------------
const int N = 80;
const double G = 1E-1;
const int DISPLAYINTERVAL = 3;
const double SPF = 0.003;        // 0.3E-2
const double RADIUS = 500;
const int TRAIL = 1;             // number of frames to “remember”
const double EPSILON = 1E0;

const int WIDTH = 1920;
const int HEIGHT = 1080;
const SDL_Color BLACK = {0, 0, 0, 255};
const SDL_Color WHITE = {255, 255, 255, 255};

// Global camera parameters
// (CAM is the “screen–center” in projection; TRANS and ROT are used in the camera’s extrinsic parameters)
Eigen::Vector3d CAM(WIDTH / 2.0, HEIGHT / 2.0, 0);
double F = 1000;  // focal length
Eigen::Vector3d TRANS(0, 0, 200);
Eigen::Vector3d ROT(0, 0, 0);

// SDL globals
SDL_Window* window = nullptr;
SDL_Renderer* renderer = nullptr;
TTF_Font* font = nullptr;

// ---------------- Helper functions ----------------

// Draw text using TTF. (If no font is available, nothing is drawn.)
void drawText(int x, int y, const std::string &text, SDL_Color color)
{
    if (!font)
        return;
    SDL_Surface* surf = TTF_RenderText_Solid(font, text.c_str(), color);
    if (!surf)
        return;
    SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surf);
    SDL_Rect dest{ x, y, surf->w, surf->h };
    SDL_RenderCopy(renderer, texture, nullptr, &dest);
    SDL_FreeSurface(surf);
    SDL_DestroyTexture(texture);
}

// Returns a scale factor (used to set circle radius) based on an object’s “depth”
double get_scale(double R, double Z)
{
    double scale = R / ((Z - CAM[2]) * (Z - CAM[2]));
    if (scale < 1)
        scale = 1;
    if (scale > 100)
        scale = 100;
    return scale;
}

// Handle keyboard events – updates the camera translation and rotation.
void handleEvent(const SDL_Event &event)
{
    if (event.type == SDL_KEYDOWN)
    {
        if (event.key.keysym.sym == SDLK_b)
        {
            CAM = Eigen::Vector3d(WIDTH / 2.0, HEIGHT / 2.0, 0);
            TRANS = Eigen::Vector3d(0, 0, 100);
            ROT = Eigen::Vector3d(0, 0, 0);
        }
    }
    // Use the current keyboard state for continuous input.
    const Uint8* keystate = SDL_GetKeyboardState(nullptr);
    if (keystate[SDL_SCANCODE_W])
        TRANS[2] -= 0.6;
    if (keystate[SDL_SCANCODE_S])
        TRANS[2] += 0.6;
    if (keystate[SDL_SCANCODE_A])
        TRANS[0] += 0.6;
    if (keystate[SDL_SCANCODE_D])
        TRANS[0] -= 0.6;
    if (keystate[SDL_SCANCODE_Q])
        TRANS[1] -= 0.6;
    if (keystate[SDL_SCANCODE_E])
        TRANS[1] += 0.6;
    if (keystate[SDL_SCANCODE_K])
        ROT[0] -= M_PI * 0.01;
    if (keystate[SDL_SCANCODE_I])
        ROT[0] += M_PI * 0.01;
    if (keystate[SDL_SCANCODE_L])
        ROT[1] += M_PI * 0.01;
    if (keystate[SDL_SCANCODE_J])
        ROT[1] -= M_PI * 0.01;
    if (keystate[SDL_SCANCODE_U])
        ROT[2] -= M_PI * 0.01;
    if (keystate[SDL_SCANCODE_O])
        ROT[2] += M_PI * 0.01;
}

// Display FPS, dummy CPU/GPU usage and tickrate information (rendered using TTF)
void fpsCounter(double tickrate, int displayFps)
{
    drawText(static_cast<int>(WIDTH / 2.2), 0, "Mode: CPU", {0, 0, 255, 255});
    // (CPU usage – not computed here – just a dummy value)
    int cpu_usage = 30;
    SDL_Color cpuColor = (cpu_usage > 60) ? SDL_Color{255, 0, 0, 255} :
                         (cpu_usage > 20 ? SDL_Color{255, 255, 0, 255} : SDL_Color{0, 255, 0, 255});
    drawText(static_cast<int>(WIDTH / 2.2), 20, "CPU: " + std::to_string(cpu_usage) + "%", cpuColor);
    drawText(static_cast<int>(WIDTH / 2.2), 40, "GPU: DummyGPU", {0, 0, 255, 255});
    SDL_Color fpsColor = (displayFps < 15) ? SDL_Color{255, 0, 0, 255} :
                         (displayFps < 30 ? SDL_Color{255, 255, 0, 255} : SDL_Color{0, 255, 0, 255});
    drawText(static_cast<int>(WIDTH / 2.2), 60, "Display FPS: " + std::to_string(displayFps), fpsColor);
    SDL_Color tickColor = (tickrate < 30) ? SDL_Color{255, 0, 0, 255} :
                          (tickrate < 60 ? SDL_Color{255, 255, 0, 255} : SDL_Color{0, 255, 0, 255});
    drawText(static_cast<int>(WIDTH / 2.2), 80, "Engine tickrate: " + std::to_string(static_cast<int>(tickrate)), tickColor);
}

// ---------------- Matrix/Physics functions ----------------

// Return the 3x3 rotation matrix for the given Euler angles.
Eigen::Matrix3d get_rot_matx(double psi, double theta, double phi)
{
    Eigen::Matrix3d R;
    R(0, 0) = cos(theta) * cos(phi);
    R(0, 1) = sin(psi) * sin(theta) * cos(phi) - cos(psi) * sin(phi);
    R(0, 2) = cos(psi) * sin(theta) * cos(phi) + sin(psi) * sin(phi);
    R(1, 0) = cos(theta) * sin(phi);
    R(1, 1) = sin(psi) * sin(theta) * sin(phi) + cos(psi) * cos(phi);
    R(1, 2) = cos(psi) * sin(theta) * sin(phi) - sin(psi) * cos(phi);
    R(2, 0) = -sin(theta);
    R(2, 1) = sin(psi) * cos(theta);
    R(2, 2) = cos(psi) * cos(theta);
    return R;
}

// Project 3D object coordinates (each row is (x,y,z)) into 2D screen coordinates.
// The projection uses a simple pinhole model: X_ = K * E * X, where X is in homogeneous form.
Eigen::MatrixXd proj_3_to_2(const Eigen::MatrixXd &obj_coord, double f,
                            const Eigen::Vector3d &trans, const Eigen::Vector3d &rot_ang)
{
    int n = static_cast<int>(obj_coord.rows());
    Eigen::Matrix3d r = get_rot_matx(rot_ang[0], rot_ang[1], rot_ang[2]);

    // Build homogeneous coordinates (n x 4)
    Eigen::MatrixXd X(n, 4);
    X.block(0, 0, n, 3) = obj_coord;
    X.col(3) = Eigen::VectorXd::Ones(n);
    Eigen::MatrixXd X_t = X.transpose(); // (4 x n)

    // Intrinsic matrix K
    Eigen::Matrix3d K;
    K << f, 0, CAM[0],
         0, f, CAM[1],
         0, 0, 1;
    // Extrinsic matrix E (3x4)
    Eigen::Matrix<double, 3, 4> E;
    E.block<3, 3>(0, 0) = r;
    E.col(3) = trans;

    Eigen::MatrixXd X_ = K * E * X_t; // (3 x n)
    Eigen::MatrixXd proj(n, 3);
    for (int i = 0; i < n; i++)
    {
        double lam = X_(2, i);
        proj(i, 0) = X_(0, i) / lam;
        proj(i, 1) = X_(1, i) / lam;
        proj(i, 2) = lam;
    }
    return proj;
}

// A simple “decay” function used in the merger process.
double decay(double M1, double M2, double v, double r)
{
    return 1E-14 * v * ((M1 * M2) / (r * r));
}

// Check and update the merger of two special particles (at indices N/4 and 3N/4).
void check_merger(Eigen::MatrixXd &x, Eigen::MatrixXd &v, Eigen::VectorXd &M, Eigen::MatrixXi &color)
{
    int idx1 = N / 4;
    int idx2 = 3 * N / 4;
    double r_ = (x.row(idx1) - x.row(idx2)).norm();
    if (r_ <= SPF * 1E2)
    {
        M(idx1) += M(idx2);
        M(idx2) = 0;
        v.row(idx1).setZero();
        v.row(idx2) *= 1E4;
        x.row(idx1) = (x.row(idx1) + x.row(idx2)) / 2.0;
        color.row(idx2).setZero();
    }
    else
    {
        double factor1 = decay(M(idx1), M(idx2), v.row(idx1).norm(), r_);
        double factor2 = decay(M(idx2), M(idx1), v.row(idx2).norm(), r_);
        v.row(idx1) = v.row(idx1) - factor1 * v.row(idx1);
        v.row(idx2) = v.row(idx2) - factor2 * v.row(idx2);
    }
}

// Update the compute_matrix (which holds positions, velocities, and accelerations) using Newtonian gravity.
// The compute_matrix is (N x 9) with columns:
// [0–2]: positions, [3–5]: velocities, [6–8]: accelerations.
void newtonian_gravitational_dynamics(Eigen::MatrixXd &compute_matrix,
                                      const std::vector<int> &/*counter*/,
                                      Eigen::VectorXd &M, Eigen::MatrixXi &color)
{
    // Extract positions x and velocities v.
    Eigen::MatrixXd x = compute_matrix.block(0, 0, N, 3);
    Eigen::MatrixXd v = compute_matrix.block(0, 3, N, 3);
    Eigen::MatrixXd a = Eigen::MatrixXd::Zero(N, 3);

    // For each particle compute the acceleration by summing the (vector) contributions
    // from every other particle.
    for (int i = 0; i < N; i++)
    {
        Eigen::Vector3d a_i = Eigen::Vector3d::Zero();
        for (int j = 0; j < N; j++)
        {
            if (i == j)
                continue;
            Eigen::Vector3d diff = x.row(i) - x.row(j);
            double r = diff.norm() + EPSILON;
            a_i += (-G * M(j) * diff) / ((r * r + EPSILON) * r);
        }
        a.row(i) = a_i;
    }

    // Update velocities and positions
    v = v + a * SPF;
    x = x + v * SPF;

    // For two special indices, perform a merger update.
    if (M(3 * N / 4) > 0)
        check_merger(x, v, M, color);

    // Update the compute_matrix with the new positions, velocities, and accelerations.
    compute_matrix.block(0, 0, N, 3) = x;
    compute_matrix.block(0, 3, N, 3) = v;
    compute_matrix.block(0, 6, N, 3) = a;
}

// ---------------- FixedQueueTensor class ----------------
// This class simply holds a fixed‐size “queue” (implemented as a vector) of frames,
// each frame being an (N x 9) matrix. The getTensor() method “stacks” the frames vertically.
class FixedQueueTensor {
public:
    int max_size;
    std::vector<Eigen::MatrixXd> queue;
    FixedQueueTensor(int max_size_ = TRAIL) : max_size(max_size_) {}
    void append(const Eigen::MatrixXd &element)
    {
        queue.push_back(element);
        if (static_cast<int>(queue.size()) > max_size)
            queue.erase(queue.begin());
    }
    Eigen::MatrixXd getTensor() const
    {
        int total_rows = N * static_cast<int>(queue.size());
        Eigen::MatrixXd stacked(total_rows, 9);
        for (size_t i = 0; i < queue.size(); i++)
        {
            stacked.block(i * N, 0, N, 9) = queue[i];
        }
        return stacked;
    }
};

// ---------------- Drawing function ----------------
// This function “renders” the scene: it projects the (stacked) compute_matrix
// (each row is a 3D point) into 2D screen space and draws a filled circle for each.
// (For simplicity the trail‐drawing is not fully implemented.)
void draw_window(const Eigen::MatrixXd &compute_matrix, const Eigen::MatrixXi &color,
                 SDL_Color lighting, double tickrate, int displayFps)
{
    // Clear the screen with the background color.
    SDL_SetRenderDrawColor(renderer, lighting.r, lighting.g, lighting.b, lighting.a);
    SDL_RenderClear(renderer);

    // The input compute_matrix is assumed to be (nframes*N x 9).
    // For projection we use the first three columns (positions).
    Eigen::MatrixXd obj_coord = compute_matrix.block(0, 0, compute_matrix.rows(), 3);
    Eigen::MatrixXd proj = proj_3_to_2(obj_coord, F, TRANS, ROT);

    // Sort the rows in descending order of the third coordinate (lam).
    std::vector<int> indices(proj.rows());
    for (int i = 0; i < proj.rows(); i++)
        indices[i] = i;
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return proj(a, 2) > proj(b, 2);
    });

    // For each projected point, if it is in front of the camera, draw a circle.
    // (Note: here we use only the “current” frame’s colors; trail‐drawing is omitted for brevity.)
    for (int idx : indices)
    {
        if (proj(idx, 2) > CAM[2])
        {
            int x = static_cast<int>(proj(idx, 0));
            int y = static_cast<int>(proj(idx, 1));
            int particleIdx = idx % N; // assume the last frame is current
            SDL_Color col;
            col.r = static_cast<Uint8>(color(particleIdx, 0));
            col.g = static_cast<Uint8>(color(particleIdx, 1));
            col.b = static_cast<Uint8>(color(particleIdx, 2));
            col.a = 255;
            double scale = get_scale(RADIUS, proj(idx, 2));
            filledCircleRGBA(renderer, x, y, static_cast<Sint16>(scale),
                             col.r, col.g, col.b, 255);
        }
    }

    fpsCounter(tickrate, displayFps);
    SDL_RenderPresent(renderer);
}

// ---------------- Main simulation ----------------
int main(int argc, char* argv[])
{
    // Initialize SDL2 and TTF.
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
    {
        std::cerr << "SDL_Init Error: " << SDL_GetError() << "\n";
        return 1;
    }
    if (TTF_Init() != 0)
    {
        std::cerr << "TTF_Init Error: " << TTF_GetError() << "\n";
        return 1;
    }
    window = SDL_CreateWindow("3D Gravity Simulator Efficient CPU C++",
                              SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                              WIDTH, HEIGHT, SDL_WINDOW_SHOWN);
    if (!window)
    {
        std::cerr << "SDL_CreateWindow Error: " << SDL_GetError() << "\n";
        return 1;
    }
    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer)
    {
        std::cerr << "SDL_CreateRenderer Error: " << SDL_GetError() << "\n";
        return 1;
    }
    // Open a font (make sure "Arial.ttf" is available in your working directory).
    font = TTF_OpenFont("./Arial.ttf", 18);
    if (!font)
        std::cerr << "TTF_OpenFont Error: " << TTF_GetError() << "\n";

    bool run = true;
    SDL_Event event;

    // Set up random generators.
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> massDist(10, 30);

    // Initialize masses (N x 1)
    Eigen::VectorXd M(N);
    for (int i = 0; i < N; i++)
        M(i) = massDist(rng) * 500;

    // Initialize position components.
    Eigen::VectorXd widthVec(N), depthVec(N), heightVec(N);
    std::uniform_int_distribution<int> widthDist1(-30, -25);
    std::uniform_int_distribution<int> widthDist2(25, 30);
    std::uniform_int_distribution<int> depthDist(-2, 2);
    std::uniform_int_distribution<int> heightDist1(-100, -95);
    std::uniform_int_distribution<int> heightDist2(95, 100);
    for (int i = 0; i < N / 2; i++)
    {
        widthVec(i) = widthDist1(rng);
        depthVec(i) = depthDist(rng);
        heightVec(i) = heightDist1(rng);
    }
    for (int i = N / 2; i < N; i++)
    {
        widthVec(i) = widthDist2(rng);
        depthVec(i) = depthDist(rng);
        heightVec(i) = heightDist2(rng);
    }

    // Initialize velocities.
    Eigen::VectorXd vx(N), vz(N), vy(N);
    std::uniform_int_distribution<int> velDist(-2, 2);
    for (int i = 0; i < N; i++)
    {
        vx(i) = velDist(rng);
        vz(i) = velDist(rng);
    }
    std::uniform_int_distribution<int> vyDist1(40, 50);
    std::uniform_int_distribution<int> vyDist2(-50, -40);
    for (int i = 0; i < N / 2; i++)
        vy(i) = vyDist1(rng);
    for (int i = N / 2; i < N; i++)
        vy(i) = vyDist2(rng);

    // Initialize color for each particle (N x 3).
    Eigen::MatrixXi colorMat(N, 3);
    for (int i = 0; i < N; i++)
    {
        int c1 = std::min(std::max(int(100 + std::pow(1.00034, M(i))), 0), 200);
        int c2 = std::min(std::max(int(-150 + std::pow(1.00037, M(i))), 0), 175);
        int c3 = std::min(std::max(int(-50 + std::pow(1.0003772, M(i))), 0), 225);
        colorMat(i, 0) = c1;
        colorMat(i, 1) = c2;
        colorMat(i, 2) = c3;
    }

    // For a few random indices, modify masses and colors.
    int numIndices = N / 20;
    std::uniform_int_distribution<int> indexDist(0, N - 1);
    std::uniform_int_distribution<int> rand100(1, 100);
    for (int i = 0; i < numIndices; i++)
    {
        int idx = indexDist(rng);
        M(idx) = 70 * rand100(rng);
        colorMat.row(idx).setConstant(255);
    }
    int idx1 = N / 4;
    int idx2 = 3 * N / 4;
    M(idx1) = 8E4 * 10;
    M(idx2) = 7.7E4 * 10;
    colorMat.row(idx1).setZero();
    colorMat.row(idx2).setZero();
    colorMat(idx1, 1) = 255;
    colorMat(idx2, 1) = 255;

    widthVec(idx1) = -10;
    widthVec(idx2) = 10;
    depthVec(idx1) = 0;
    depthVec(idx2) = 0;
    heightVec(idx1) = -70;
    heightVec(idx2) = 70;
    vy(idx1) = 5;
    vy(idx2) = -5;

    // Build the compute_matrix (N x 9): columns 0–2: positions, 3–5: velocities, 6–8: acceleration (initially zero)
    Eigen::MatrixXd compute_matrix(N, 9);
    compute_matrix.col(0) = widthVec;
    compute_matrix.col(1) = heightVec;
    compute_matrix.col(2) = depthVec;
    compute_matrix.col(3) = vx;
    compute_matrix.col(4) = vy;
    compute_matrix.col(5) = vz;
    compute_matrix.col(6).setZero();
    compute_matrix.col(7).setZero();
    compute_matrix.col(8).setZero();

    std::cout << "compute_matrix shape: " << compute_matrix.rows() << "x" << compute_matrix.cols() << std::endl;

    int time_step = 1;
    SDL_Color lighting = BLACK;
    std::vector<int> counter(N);
    for (int i = 0; i < N; i++)
        counter[i] = i;

    auto start_time = std::chrono::high_resolution_clock::now();
    FixedQueueTensor logQueue(TRAIL);
    logQueue.append(compute_matrix);

    // Main simulation loop
    while (run)
    {
        while (SDL_PollEvent(&event))
        {
            if (event.type == SDL_QUIT)
                run = false;
            handleEvent(event);
        }

        newtonian_gravitational_dynamics(compute_matrix, counter, M, colorMat);

        if (time_step % DISPLAYINTERVAL == 0)
        {
            // Update the camera (rotate and translate slightly).
            ROT[1] += M_PI * 0.03;
            TRANS[2] -= 0.001;

            logQueue.append(compute_matrix);
            auto current_time = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(current_time - start_time).count();
            int displayFps = (elapsed > 0) ? int(1.0 / elapsed) : 0;
            double tickrate = DISPLAYINTERVAL * displayFps;
            start_time = current_time;

            Eigen::MatrixXd stacked = logQueue.getTensor();
            draw_window(stacked, colorMat, lighting, tickrate, displayFps);
            SDL_Delay(1);  // small delay to avoid hogging the CPU
        }
        time_step++;
    }

    // Clean up and quit.
    if (font)
        TTF_CloseFont(font);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    TTF_Quit();
    SDL_Quit();

    return 0;
}

