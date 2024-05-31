#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include </usr/include/eigen3/Eigen/Dense> 
#include <tuple>

namespace py = pybind11;

using namespace std;


//cambiar los vectores por matrices de Eigen

float norma1 (Eigen::VectorXf q1, Eigen::VectorXf q2){
    float count = 0;
    for(int i = 0; i < q1.size(); i++){
        count += abs(q1(i) - q2(i));
    }
    return count;
}

Eigen::MatrixXf conseguirMatriz(const string &file){
    ifstream source;
    source.open(file);
    string linea;
    Eigen::MatrixXf matriz;
    float number;
    int i = 0;

    while(getline(source, linea)){
        Eigen::VectorXf fila;
        stringstream iss(linea);
        int j = 0;
        while(iss >> number){
            fila.conservativeResize(fila.size()+1);
            fila(i) = number;
            j++;
        }
        matriz.conservativeResize(matriz.rows()+1, matriz.cols());
        matriz.row(i) = fila;
        i++;
    }

    source.close();
    return matriz;
}

tuple<float, Eigen::VectorXf> metodoPotencia(const Eigen::MatrixXf &matriz, const float tolerancia, int iteraciones){
    int n = matriz.rows();
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
    return make_tuple(autovalor, q);
}

// Función para aplicar la transformación de Householder
Eigen::MatrixXf householder(const Eigen::VectorXf& v) {
    int n = v.size();
    Eigen::MatrixXf H = Eigen::MatrixXf::Identity(n, n);
    Eigen::VectorXf w = v;

    float alpha = v.norm();
    if (v(0) < 0) {
        alpha = -alpha;
    }
    w(0) += alpha;

    float beta = 2 / w.squaredNorm();
    H -= beta * (w * w.transpose());

    return H;
}

PYBIND11_MODULE(my_module, m) {
    m.doc() = "aplicacion metodo de la potencia"; // Documentación opcional
    m.def("metodoPotencia", &metodoPotencia, "Power method for computing eigenvalues");
    m.def("householder", &householder, "Householder transformation function");
}