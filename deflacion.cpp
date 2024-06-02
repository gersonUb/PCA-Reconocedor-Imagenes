#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include </usr/include/eigen3/Eigen/Dense> 
#include <tuple>

using namespace std;


vector<tuple<float, Eigen::VectorXf>> metodoPotencia(const Eigen::MatrixXf &matriz, const float tolerancia, int iteraciones){

    vector<tuple<float, Eigen::VectorXf>> resultado;
    int n = matriz.rows();
    for (int i = 0; i < n; ++i) {
        
        Eigen::VectorXf q(n);
        for(int i = 0; i < n; i++){
            q(i) = rand();
        }
        q.normalize();
        Eigen::VectorXf q_anterior(n);
        q_anterior = Eigen::VectorXf::Zero(n);

        while(iteraciones > 0 ){
            q_anterior = q;
            q = matriz * q_anterior;
            q.normalize();
            iteraciones--;
        }

        float autovalor = float((q.transpose()) * matriz * q) / float((q.transpose() * q));

    
        auto par = make_tuple(autovalor, q);
        resultado.push_back(par);

        // Deflación: actualizamos la matriz A
        A = A - autovalor * (q * q.transpose());
    }

   return resultado;
}
