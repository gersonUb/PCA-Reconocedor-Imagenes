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

tuple<vector<float>, vector<Eigen::VectorXf>, vector<int>> metodoPotenciaDeflacion(const Eigen::MatrixXf &matriz, const float tolerancia, int max_iteraciones) {
    vector<float> autovalores;
    vector<Eigen::VectorXf> autovectores;
    vector<int> iteraciones_totales;

    int n = matriz.rows();
    vector<Eigen::MatrixXf> matrices;
    matrices.push_back(matriz);

    for (int i = 0; i < n; ++i) {
        Eigen::MatrixXf matriz_auxiliar = matrices[i];
        Eigen::VectorXf q = Eigen::VectorXf::Random(n);
        q.normalize();
        Eigen::VectorXf q_anterior = Eigen::VectorXf::Zero(n);

        int iteraciones = max_iteraciones;
        int contador = 0;
        while (iteraciones > 0 && (q - q_anterior).cwiseAbs().maxCoeff() >= tolerancia) {
            q_anterior = q;
            q = matriz_auxiliar * q_anterior;
            q.normalize();
            iteraciones--;
            contador++;
        }

        float autovalor = float((q.transpose() * matriz_auxiliar * q)) / float((q.transpose() * q));

        autovalores.push_back(autovalor);
        autovectores.push_back(q);
        iteraciones_totales.push_back(contador);

        // Deflación: actualizamos la matriz
        matrices.push_back(matriz_auxiliar - autovalor * (q * q.transpose()));
    }
    return make_tuple(autovalores, autovectores, iteraciones_totales);
}

int main() {
    // Definir la matriz de prueba
    Eigen::MatrixXf matriz(2, 2);
    matriz << 1, 1,
              1, 1;

    // Definir la tolerancia y el número máximo de iteraciones
    float tolerancia = 0.0001;
    int iteraciones = 1000000;

    auto resultados = metodoPotenciaDeflacion(matriz, tolerancia, iteraciones);

    // Imprimir los resultados para cada autovalor y autovector
    vector<float> autovalores;
    vector<Eigen::VectorXf> autovectores;
    vector<int> iteraciones_totales;

    tie(autovalores, autovectores, iteraciones_totales) = resultados;

    for (size_t i = 0; i < autovalores.size(); ++i) {
        cout << "Resultado " << i + 1 << ":" << endl;
        cout << "Autovalor: " << autovalores[i] << endl;
        cout << "Autovector: " << autovectores[i].transpose() << endl;
        cout << "Cantidad de pasos: " << iteraciones_totales[i] << endl;
        cout << "----------------------" << endl;
    }

    return 0;
}
