tuple<std::vector<float>, std::vector<Eigen::VectorXf>> encontrarTodosAutovalAutovec(const Eigen::MatrixXf &matriz, const float tolerancia, int iteraciones) {
    Eigen::MatrixXf A = matriz;
    int n = matriz.rows();
    std::vector<float> autovalores;
    std::vector<Eigen::VectorXf> autovectores;

    for (int i = 0; i < n; ++i) {
        auto [autovalor, autovector] = metodoPotencia(A, tolerancia, iteraciones);
        autovalores.push_back(autovalor);
        autovectores.push_back(autovector);

        // Deflación: actualizamos la matriz A
        A = A - autovalor * (autovector * autovector.transpose());
    }

    return make_tuple(autovalores, autovectores);
}
