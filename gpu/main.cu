#include <iostream>
#include <vector>
#include <set>
#include <limits>
#include <stdlib.h>
#include <ctime>
#include <thread>
#include <iomanip>
#include <sys/time.h>
//#define INF std::numeric_limits<int>::max()
#define INF 1147483647

template<std::size_t n>
__device__ void dijkstra(int *graph, int graphIdx, int source, int target, int thread, bool* solved);

template<std::size_t n>
__device__ void Astar(int *graph, int graphIdx, int source, int target, int thread, bool* solved);

template<std::size_t n>
void randomGraph(int (&graph)[n][n]);

template<std::size_t n>
void addCloseBias(int (&graph)[n][n]);

#define NUM_GRAPHS 2048

void stop(std::thread ts[NUM_GRAPHS])
{
  for(int i = 0; i < NUM_GRAPHS; ++i)
  {
    ts[i].join();
  }
}

int bg[NUM_GRAPHS][500][500];
int ss[NUM_GRAPHS];
int ts[NUM_GRAPHS];
__host__ __device__ inline int index(const int x, const int y, const int z) {
     return x * 500 * 500 + y * 500 + z;
}

__global__ void pathfind(int *g, int ts[NUM_GRAPHS], int ss[NUM_GRAPHS], bool *solved)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if(idx < NUM_GRAPHS)
  {
    int i = idx/2;
    if(idx % 2 == 0)
      Astar<500>(g, i, ss[i], ts[i], i, solved);
    else
      dijkstra<500>(g, i, ss[i], ts[i], i, solved);
  }
}
__global__ void d(int *g, int ts[NUM_GRAPHS], int ss[NUM_GRAPHS], bool *solved)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if(idx < NUM_GRAPHS)
  {
    dijkstra<500>(g, idx, ss[idx], ts[idx], idx, solved);
  }
}

__global__ void a(int *g, int ts[NUM_GRAPHS], int ss[NUM_GRAPHS], bool *solved)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if(idx < NUM_GRAPHS)
  { 
    Astar<500>(g, idx, ss[idx], ts[idx], idx, solved);
  }
}


#define THREADS_PER_BLOCK 1000
int main(int argc, char* argv[]) 
{
  srand(time(NULL));
  bool* solved = (bool*) malloc(NUM_GRAPHS*sizeof(bool));
  for(int i = 0; i < NUM_GRAPHS; ++i)
  {
    solved[i] = false;
    randomGraph(bg[i]);
    addCloseBias(bg[i]);
    ss[i] = 0 + (rand() % (int)(499 - 0 + 1));
    ts[i] = 0 + (rand() % (int)(499 - 0 + 1));
  }
  bool* solved_gpu;
  cudaMalloc(&solved_gpu, NUM_GRAPHS*sizeof(bool));
  cudaMemcpy(solved_gpu, solved, NUM_GRAPHS*sizeof(bool), cudaMemcpyHostToDevice);
  
  int *flatG = (int*) malloc(NUM_GRAPHS*500*500*sizeof(int));
  for(int x = 0; x < NUM_GRAPHS; ++x)
    for(int y = 0; y < 500; ++y)
      for(int z = 0; z < 500; ++z)
        flatG[x*500*500 + y*500 + z] = bg[x][y][z];
  
  int *g;
  cudaMalloc(&g, NUM_GRAPHS*500*500*sizeof(int));
  cudaMemcpy(g, flatG, NUM_GRAPHS*500*500*sizeof(int), cudaMemcpyHostToDevice);
  
  int *ss_gpu;
  cudaMalloc(&ss_gpu, NUM_GRAPHS*sizeof(int));
  cudaMemcpy(ss_gpu, ss, NUM_GRAPHS*sizeof(int), cudaMemcpyHostToDevice);
  
  int *ts_gpu;
  cudaMalloc(&ts_gpu, NUM_GRAPHS*sizeof(int));
  cudaMemcpy(ts_gpu, ts, NUM_GRAPHS*sizeof(int), cudaMemcpyHostToDevice);
  
  struct timeval startTime, endTime;
  
  gettimeofday(&startTime, 0);
  
  int numBlocks = NUM_GRAPHS*2/THREADS_PER_BLOCK;
  //pathfind<<<numBlocks, THREADS_PER_BLOCK>>>(g, ss_gpu, ts_gpu, solved_gpu);
  bool astarf = argv[1][0] == 'a';
  bool dijf = argv[1][0] == 'd';
  bool bothf = argv[1][0] == 'b';

  cudaStream_t streams[2];
  if(dijf || bothf)
    cudaStreamCreate(&streams[0]);
  if(astarf || bothf)
    cudaStreamCreate(&streams[1]);
  
  if(dijf || bothf)
    d<<<numBlocks/2, THREADS_PER_BLOCK, 0, streams[0]>>>(g, ss_gpu, ts_gpu, solved_gpu);
  if(astarf || bothf)
    a<<<numBlocks/2, THREADS_PER_BLOCK, 0, streams[1]>>>(g, ss_gpu, ts_gpu, solved_gpu); 
  
  //sync to see print statements
  cudaDeviceSynchronize();
  gettimeofday(&endTime, 0);
  std::cout << (1000000.0*(endTime.tv_sec-startTime.tv_sec) + endTime.tv_usec-startTime.tv_usec)/1000.0 << " ms"<< std::endl;
  return 0;
}

__device__ bool setEmpty(bool* set, int size)
{
  for(int i = 0; i < size; ++i)
    if(set[i])
      return false;
  return true;
}

__device__ int firstSet(bool*set, int size)
{
  for(int i = 0; i < size; ++i)
    if(set[i])
      return i;
  return -1;
}

template<std::size_t n>
__device__ void dijkstra(int *graph, int graphIdx, int source, int target, int thread, bool* solved)
{
  int *dist = new int[n];
  for(int i = 0; i < n; ++i)
    dist[i] = INF;
  dist[source] = 0;
  bool *vertices = new bool[n];
  for(int i = 0; i < n; i++)
  {
    vertices[i] = true;
  }
  while(!setEmpty(vertices, n))
  {
    if(solved[thread])
    {
      return;
    }
    int minV = firstSet(vertices, n);
    int min = dist[minV];
    for(int i = 0; i < n; ++i)
    {
      if(vertices[i] && dist[i] < min)
      {
        minV = i;
        min = dist[i];
      }
    }
    if(minV == target)
      break;
    vertices[minV] = false;
    for(int i = 0; i < n; ++i)
    {
      //skip if not adjacent
      if(graph[index(graphIdx,minV,i)] == INF)
        continue;
      int newDist = dist[minV] + graph[index(graphIdx,minV,i)];
      if(newDist < dist[i])
      {
        dist[i] = newDist;
      }
    }
  }
  solved[thread] = true;
  printf("d:%d\n", dist[target]);
}
#include <cmath>
template<std::size_t n>
__device__ void Astar(int *graph, int graphIdx, int source, int target, int thread, bool* solved)
{
  int *dist = new int[n];
  for(int i = 0; i < n; ++i)
    dist[i] = INF;
  dist[source] = 0;
  bool *vertices = new bool[n];
  for(int i = 0; i < n; i++)
  {
    vertices[i] = true;
  }
  while(!setEmpty(vertices, n))
  {
    if(solved[thread])
    {
      return;
    }
    int minV = firstSet(vertices, n);
    int min = dist[minV];
    for(int i = 0; i < n; ++i)
    {
      if(vertices[i] && dist[i]+2*(std::abs(i-target)) < min)
      {
        minV = i;
        min = dist[i];
      }
    }
    if(minV == target)
      break;
    vertices[minV] = false;
    for(int i = 0; i < n; ++i)
    {
      //skip if not adjacent
      if(graph[index(graphIdx,minV,i)] == INF)
        continue;
      int newDist = dist[minV] + graph[index(graphIdx,minV,i)];
      if(newDist < dist[i])
      {
        dist[i] = newDist;
      }
    }
  }
  solved[thread] = true;
  printf("d:%d\n", dist[target]);
}

template<std::size_t n>
void randomGraph(int (&graph)[n][n])
{
  int maxEdges = n*(n-1);
  int numEdges = n + (rand() % (int)(maxEdges - (n) + 1));
  for(int i = 0; i < n; ++i)
    for(int j = 0; j < n; ++j)
      graph[i][j] = INF;
  
  for(int i = 0; i < numEdges; ++i)
  {
    int v1 = (rand() % (int)(n));
    int v2 = (rand() % (int)(n));
    //don't allow loops
    if(v1 == v2)
    {
      i--;
      continue;
    }
    int weight = 1 + (rand() % (int)(1000 - 1 + 1));
    graph[v1][v2] = graph[v2][v1] = weight;
  }
}

template<std::size_t n>
void addCloseBias(int (&graph)[n][n])
{
  for(int v = 0; v < n-10; )
  {
    int w1 = 1 + (rand() % (int)(5 - 1 + 1));
    int w2 = 1 + (rand() % (int)(5 - 1 + 1));
    int w3 = 1 + (rand() % (int)(5 - 1 + 1));
    int v1 = (v+1) + (rand() % (int)((v + 10) - (v+1) + 1));
    int v2 = (v+2) + (rand() % (int)((v + 10) - (v+2) + 1));
    int v3 = (v+3) + (rand() % (int)((v + 10) - (v+3) + 1));
    graph[v][v1] = graph[v1][v] = w1;
    graph[v][v2] = graph[v2][v] = w2;
    graph[v][v3] = graph[v3][v] = w3;
    v += 1 + (rand() % (int)(5 - 1 + 1)); 
    if(v >= n)
      break;
  }  
}
