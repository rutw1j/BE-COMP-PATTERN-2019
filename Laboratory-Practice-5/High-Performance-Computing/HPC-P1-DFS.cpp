    #include <iostream>
    #include <vector>
    #include <stack>
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

            void ParallelDFS(int start) {
                vector<bool> visited(V, false);
                stack<int> s;

                s.push(start);

                while(!s.empty()) {
                    int current = s.top();
                    s.pop();
                    
                    if (!visited[current]) {
                        cout << current << " -> ";
                        visited[current] = true;

                        #pragma omp parallel for
                        for ( int i = 0; i < adj[current].size(); ++i ) {
                            int neighbour = adj[current][i];
                            if (!visited[neighbour]) {
                                s.push(neighbour);
                            }
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

        cout << "\nDFS TRAVERSA\nL";
        graph.ParallelDFS(0);
        cout << "End";
    }