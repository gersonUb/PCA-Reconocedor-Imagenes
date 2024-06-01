#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include </usr/include/eigen3/Eigen/Dense> 
#include <tuple>
#include <ctime>

using namespace std;


tuple<float, Eigen::VectorXf, int> metodoPotencia(const Eigen::MatrixXf &matriz, const float tolerancia, int iteraciones){
    int n = matriz.rows();
    Eigen::VectorXf q(n);
    for(int i = 0; i < n; i++){
        q(i) = rand();
    }
    q.normalize();
    Eigen::VectorXf q_anterior(n);
    q_anterior = Eigen::VectorXf::Zero(n);
    int contador = 0;

    while(iteraciones > 0 and ((q-q_anterior).cwiseAbs()).maxCoeff() >= tolerancia ){
        q_anterior = q;
        q = matriz * q_anterior;
        q.normalize();
        iteraciones--;
        contador ++;
    }

    float autovalor = float((q.transpose()) * matriz * q) / float((q.transpose() * q));
    return make_tuple(autovalor, q, contador);
}


tuple<float, Eigen::VectorXf, int> metodoPotenciaDeflacion(const Eigen::MatrixXf &matriz, const float tolerancia, int iteraciones, float autovalor ,const Eigen::VectorXf &autovector) {
    Eigen::MatrixXf matriz_deflacionada = matriz - autovalor * (autovector * autovector.transpose());
    return metodoPotencia(matriz_deflacionada, tolerancia, iteraciones);
}

int main() {
    // Definir la matriz de prueba
    Eigen::MatrixXf matriz(2, 2);
    matriz << 1, 1,
              1, 1;

    // Definir la tolerancia y el número máximo de iteraciones
    float tolerancia = 0.001;
    int iteraciones = 1000000;

    // Ejecutar el método de la potencia para el primer autovalor y autovector
    int pasos1;
    float autovalor1;
    Eigen::VectorXf autovector1;
    tie(autovalor1, autovector1, pasos1) = metodoPotencia(matriz, tolerancia, iteraciones);

    // Imprimir los resultados del primer autovalor y autovector
    cout << "Primer autovalor: " << autovalor1 << endl;
    cout << "Primer autovector: " << autovector1.transpose() << endl;
    cout << "Cantidad de pasos para el primer autovalor: " << pasos1 << endl;

    // Ejecutar el método de la potencia con deflación para el segundo autovalor y autovector
    int pasos2;
    float autovalor2;
    Eigen::VectorXf autovector2;
    tie(autovalor2, autovector2, pasos2) = metodoPotenciaDeflacion(matriz, tolerancia, iteraciones, autovalor1 ,autovector1);

    // Imprimir los resultados del segundo autovalor y autovector
    cout << "\nSegundo autovalor: " << autovalor2 << endl;
    cout << "Segundo autovector: " << autovector2.transpose() << endl;
    cout << "Cantidad de pasos para el segundo autovalor: " << pasos2 << endl;

    return 0;
}