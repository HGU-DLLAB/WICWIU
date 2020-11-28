#include <iostream>
#include <string>

#include "../../../WICWIU_src/NeuralNetwork.hpp"


class my_RNN : public NeuralNetwork<float>{
private:
public:
    my_RNN(Tensorholder<float> *x, Tensorholder<float> *label, int vocab_length, int embedding_dim) {
        SetInput(2, x, label);

        Operator<float> *out = NULL;


        //===================== Embedding Layer ==================
        out = new EmbeddingLayer<float>(x, vocab_length, embedding_dim, "Embedding");

        // ======================= layer 1=======================
        //out = new RecurrentLayer<float>(out, embedding_dim, 64, vocab_length, TRUE, "Recur_1");
        //out = new LSTM2Layer<float>(out, embedding_dim, 128, vocab_length, TRUE, "Recur_1");
        out = new GRULayer<float>(out, embedding_dim, 128, vocab_length, TRUE, "Recur_1");

        // // ======================= layer 2=======================
        // out = new Linear<float>(out, 5 * 5 * 20, 1024, TRUE, "Fully-Connected_1");

        AnalyzeGraph(out);

        // ======================= Select LossFunction Function ===================
        // SetLossFunction(new HingeLoss<float>(out, label, "HL"));
        // SetLossFunction(new MSE<float>(out, label, "MSE"));
        SetLossFunction(new SoftmaxCrossEntropy<float>(out, label, "SCE"));
        // SetLossFunction(new CrossEntropy<float>(out, label, "CE"));

        // ======================= Select Optimizer ===================
        SetOptimizer(new GradientDescentOptimizer<float>(GetParameter(), 0.001, 0.9, 1.0, MINIMIZE));
        //SetOptimizer(new RMSPropOptimizer<float>(GetParameter(), 0.01, 0.9, 1e-08, FALSE, MINIMIZE));
        //SetOptimizer(new AdamOptimizer<float>(GetParameter(), 0.001, 0.9, 0.999, 1e-08, MINIMIZE));
        // SetOptimizer(new NagOptimizer<float>(GetParameter(), 0.001, 0.9, MINIMIZE));
        //SetOptimizer(new AdagradOptimizer<float>(GetParameter(), 0.001, 0.9, MINIMIZE));      //MAXIMIZE
    }

    virtual ~my_RNN() {}
};
