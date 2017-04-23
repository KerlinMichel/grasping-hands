#include <iostream>
#include <vector>
#include <set>
#include <limits>
#include <stdlib.h>
#include <ctime>
#include <chrono>
#include <thread>
#include <iomanip>
//#define INF std::numeric_limits<int>::max()
#define INF 2147483647
#define currTime duration_cast<milliseconds>(system_clock::now().time_since_epoch());

using namespace std::chrono;

template<std::size_t n>
__global__ void dijkstra(int (&graph)[n][n], int source, int target, int thread, bool* solved);

template<std::size_t n>
void Astar(int (&graph)[n][n], int source, int target, int thread, bool* solved);

template<std::size_t n>
void randomGraph(int (&graph)[n][n]);

template<std::size_t n>
void addCloseBias(int (&graph)[n][n]);

#define NUM_GRAPHS 1000
//bool solved[NUM_GRAPHS];

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
void a(int t1, int t2, bool* solved)
{
  for(int i = t1; i < t2; ++i)
  {
    Astar(bg[i], ss[i], ts[i], i, solved);
  }

}
void d(int t1, int t2, bool* solved)
{
  for(int i = t1; i < t2; ++i)
  {
    //dijkstra<500><<<5,5>>>(bg[i], ss[i], ts[i], i, solved);
  }
}
int main() 
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
  //d(0, 1000, solved_gpu);
  std::vector<int> dist(500, INF);
  dijkstra<500><<<5,5>>>(bg[0], ss[0], ts[0], 0, solved);
  //sync to see print statements
  cudaDeviceSynchronize();
  //std::thread th1(d, 0, NUM_GRAPHS);
  //std::thread th2(a, 0, NUM_GRAPHS);
  //th1.join();
  //th2.join();
  /*std::thread a_ths[NUM_GRAPHS];
  std::thread d_ths[NUM_GRAPHS];
  for(int i = 0; i < NUM_GRAPHS; ++i)
  {
    a_ths[i] = std::thread(Astar<500>, std::ref(bg[i]), ss[i], ts[i], i);
    d_ths[i] = std::thread(dijkstra<500>, std::ref(bg[i]), ss[i], ts[i], i);
  }*/
  /*for(int i = 0; i < NUM_GRAPHS; i+=10)
  {
    a_ths[i/10] = std::thread(a, i, i+10);
    d_ths[i/10] = std::thread(d, i, i+10);
  }*/
  /*std::thread s1(stop, a_ths);
  std::thread s2(stop, d_ths);
  s1.join();
  s2.join();*/
  return 0;
}

__device__ bool inSet()
{
  return true;
}
template<std::size_t n>
__global__ void dijkstra(int (&graph)[n][n], int source, int target, int thread, bool* solved)
{
  //std::vector<int> dist(n, INF);
  int *dist = new int[n];
  for(int i = 0; i < n; ++i)
    dist[i] = INF;
  inSet();
  dist[source] = 0;// dist.at(source) = 0;
  bool *vertices = new bool[n];
  //std::set<int> vertices;
  for(int i = 0; i < n; i++)
  {
    vertices[i] = true;
    //vertices.insert(i);
  }
  printf("Test\n");
  /*while(!vertices.empty())
  {
    if(solved[thread])
    {
      return;
    }
    std::set<int>::iterator it;
    int minV = *vertices.begin();
    int min = dist[minV];//dist.at(minV);
    for(it = vertices.begin(); it != vertices.end(); ++it)
    {
      if(dist[*it] < min)//dist.at(*it) < min)
      {
        minV = *it;
        min = dist[*it];//dist.at(*it);
      }
    }
    if(minV == target)
      break;
    vertices.erase(minV);
    for(it = vertices.begin(); it != vertices.end(); ++it)
    {
      //skip if not adjacent
      if(graph[minV][*it] == INF)
        continue;
      int newDist = dist[minV] + graph[minV][*it];//dist.at(minV) + graph[minV][*it];
      if(newDist < dist[*it])//dist.at(*it))
      {
        dist[*it] = newDist;// dist.at(*it) = newDist;
      }
    }
  }*/
  solved[thread] = true;
  for(int i = 0; i < n; i++)
  {
    //std::cout << i <<": " << dist.at(i) << " | "; 
  }
}

template<std::size_t n>
void Astar(int (&graph)[n][n], int source, int target, int thread, bool* solved)
{
  std::vector<int> dist(n, INF);
  dist.at(source) = 0;
  std::set<int> vertices;
  for(int i = 0; i < n; i++)
  {
    vertices.insert(i);
  }
  while(!vertices.empty())
  {
    if(solved[thread])
    {
      return;
    }
    std::set<int>::iterator it;
    int minV = *vertices.begin();
    int min = dist.at(minV);
    for(it = vertices.begin(); it != vertices.end(); ++it)
    {
      if((dist.at(*it) + 2*(*it)) < min)
      {
        minV = *it;
        min = dist.at(*it);
      }
    }
    if(minV == target)
      break;
    vertices.erase(minV);
    for(it = vertices.begin(); it != vertices.end(); ++it)
    {
      //skip if not adjacent
      if(graph[minV][*it] == INF)
        continue;
      int newDist = dist.at(minV) + graph[minV][*it];
      if(newDist < dist.at(*it))
      {
        dist.at(*it) = newDist;
      }
    }
  }
  for(int i = 0; i < n; i++)
  {
    //std::cout << "d: " << dist.at(i) << std::endl;
  }
  solved[thread] = true;
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
