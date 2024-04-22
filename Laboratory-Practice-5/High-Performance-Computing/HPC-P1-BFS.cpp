#include<iostream>
#include<omp.h>
#include<vector>
#include<queue>

using namespace std;


class Graph {
    private:
        int V;
        vector<vector<int>> adj;

    public:
        Graph(int vertices): V(vertices), adj(vertices) {}

        void add_edge(int u, int v) {
            adj[u].push_back(v);
            adj[v].push_back(u);
        }

        void ParallelBFS(int start) {
            vector<bool> visited(V, false);
            queue<int> q;

            visited[start] = true;
            q.push(start);

            while(!q.empty()) {
                int current = q.front();
                cout << current << " -> ";
                q.pop();

                #pragma omp parallel for
                for (int i = 0; i < adj[current].size(); i++) {
                    int neighbour = adj[current][i];
                    if (!visited[neighbour]) {
                        visited[neighbour] = true;
                        q.push(neighbour);
                    }
                }
            }
        }
};


int main() {
    Graph graph(7);
    
    graph.add_edge(0, 1);
    graph.add_edge(0, 2);
    graph.add_edge(1, 3);
    graph.add_edge(1, 4);
    graph.add_edge(2, 5);
    graph.add_edge(2, 6);
    
    cout << "\nBFS TRAVERSAL\n";
    graph.ParallelBFS(0);
    cout << "End";
}