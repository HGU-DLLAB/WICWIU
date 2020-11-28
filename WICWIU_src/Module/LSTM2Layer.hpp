#ifndef __LSTM2_LAYER__
#define __LSTM2_LAYER__    value

#include "../Module.hpp"


template<typename DTYPE> class LSTM2Layer : public Module<DTYPE>{
private:
public:

    LSTM2Layer(Operator<DTYPE> *pInput, int inputsize, int hiddensize, int outputsize, int use_bias = TRUE, std::string pName = "No Name") : Module<DTYPE>(pName) {
        Alloc(pInput, inputsize, hiddensize, outputsize, use_bias, pName);
    }

    virtual ~LSTM2Layer() {}

    int Alloc(Operator<DTYPE> *pInput, int inputsize, int hiddensize, int outputsize, int use_bias, std::string pName) {
        this->SetInput(pInput);

        Operator<DTYPE> *out = pInput;


        //weight
        Tensorholder<DTYPE> *pWeight_IG = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, 4*hiddensize, inputsize, 0.0, 0.01), "LSTMLayer_pWeight_IG_" + pName);
        Tensorholder<DTYPE> *pWeight_HG = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, 4*hiddensize, hiddensize, 0.0, 0.01), "LSTMLayer_pWeight_HG_" + pName);

        //output weight
        Tensorholder<DTYPE> *pWeight_h2o = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, outputsize, hiddensize, 0.0, 0.01), "LSTMLayer_pWeight_HO_" + pName);

        //bias
        Tensorholder<DTYPE> *lstmBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, 4*hiddensize, 0.f), "RNN_Bias_f" + pName);

        out = new LSTM2<DTYPE>(out, pWeight_IG, pWeight_HG, lstmBias);

        out = new MatMul<DTYPE>(pWeight_h2o, out, "rnn_matmul_ho");



        if (use_bias) {
            Tensorholder<DTYPE> *pBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, outputsize, 0.f), "Add_Bias_" + pName);
            out = new AddColWise<DTYPE>(out, pBias, "Layer_Add_" + pName);
        }

        this->AnalyzeGraph(out);

        return TRUE;
    }
};


#endif  // __LSMT2_LAYER__
