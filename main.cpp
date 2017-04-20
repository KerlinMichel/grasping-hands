#include <iostream>
#include <vector>
#include <set>
#include <limits>
#include <stdlib.h>
#include <ctime>
#include <chrono>
#define INF std::numeric_limits<int>::max()
#define currTime duration_cast<milliseconds>(system_clock::now().time_since_epoch());

using namespace std::chrono;
int graph[5][5] = {
  { 0, 4,INF, 3, 2},
  { 4, 0, 5,INF,INF},
  {INF, 5, 0, 6, 1},
  { 3,INF, 6, 0,INF},
  { 2,INF, 1,INF, 0}
};

template<std::size_t n>
void dijkstra(int (&graph)[n][n], int source, int target);

template<std::size_t n>
void Astar(int (&graph)[n][n], int source, int target);

template<std::size_t n>
void randomGraph(int (&graph)[n][n]);

int main() 
{
  srand(time(NULL));
  int bg[100][100];
  for(int i = 0; i < 10000; ++i)
  {
    randomGraph(bg);
    dijkstra(bg, 0, 10);
    //std::cout << std::endl;
  }
  return 0;
}

template<std::size_t n>
void dijkstra(int (&graph)[n][n], int source, int target)
{
  std::vector<int> dist(n, INF);
  dist.at(source) = 0;
  std::set<int> vertices;
  for(int i = 0; i < n; i++)
  {
    vertices.insert(i);
    //dist.at(i) = INF;
    //std::cout << "d: " << dist.at(i) << std::endl;
  }
  while(!vertices.empty())
  {
    std::set<int>::iterator it;
    int minV = *vertices.begin();
    int min = dist.at(minV);
    for(it = vertices.begin(); it != vertices.end(); ++it)
    {
      if(dist.at(*it) < min)
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
    //std::cout << i <<": " << dist.at(i) << " | "; 
  }
}

template<std::size_t n>
void Astar(int (&graph)[n][n], int source, int target)
{
  std::vector<int> dist(n, INF);
  dist.at(source) = 0;
  std::set<int> vertices;
  for(int i = 0; i < n; i++)
  {
    vertices.insert(i);
    //dist.at(i) = INF;
    //std::cout << "d: " << dist.at(i) << std::endl;
  }
  while(!vertices.empty())
  {
    std::set<int>::iterator it;
    int minV = *vertices.begin();
    int min = dist.at(minV);
    for(it = vertices.begin(); it != vertices.end(); ++it)
    {
      if(dist.at(*it) < min)
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
    std::cout << "d: " << dist.at(i) << std::endl;
  }
}

template<std::size_t n>
void randomGraph(int (&graph)[n][n])
{
  int maxEdges = n*(n-1);
  int numEdges = (n/10) + (rand() % (int)(maxEdges - (n/10) + 1));
  for(int i = 0; i < n; ++i)
    for(int j = 0; j < n; ++j)
      graph[i][j] = INF;
  
  for(int i = 0; i < numEdges; ++i)
  {
    int v1 = (rand() % (int)(n));
    int v2 = (rand() % (int)(n));
    int weight = 1 + (rand() % (int)(1000 - 1 + 1));
    graph[v1][v2] = graph[v2][v1] = weight;
  }
  // don't allow loops
  for(int i = 0; i < n; ++i)
  {
    graph[i][i] == 0;
  }
  /*for(int i = 0; i < n; ++i)
  {
    for(int j = 0; j < n; ++j)
      std::cout << graph[i][j] << ",";

    std::cout << std::endl;
  }*/
}
