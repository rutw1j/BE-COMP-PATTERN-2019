#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>

using namespace std;


class Graph {

    private:
        int V;
        vector <vector<int>> adj;

    public:
        Graph(int vertices) : V(vertices), adj(vertices) {}

        void addEdge(int u, int v) {
            adj[u].push_back(v);
            adj[v].push_back(u);
        }

        void BFS(int start) {
            vector<bool> visited(V, false);
            queue<int> q;

            visited[start] = true;
            q.push(start);

            while(!q.empty()) {
                int current = q.front();
                cout << current << " -> ";
                q.pop();

                #pragma omp parallel for
                for ( int i = 0; i < adj[current].size(); i++ ) {
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

    graph.addEdge(0, 1);
    graph.addEdge(0, 2);
    graph.addEdge(1, 3);
    graph.addEdge(1, 4);
    graph.addEdge(2, 5);
    graph.addEdge(2, 6);

    cout << "BFS TRAVERSAL" << endl;
    graph.BFS(0);
    cout << "End";
}