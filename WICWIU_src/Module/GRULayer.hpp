#ifndef __GRU_LAYER__
#define __GRU_LAYER__    value

#include "../Module.hpp"


template<typename DTYPE> class GRULayer : public Module<DTYPE>{
private:
public:

    GRULayer(Operator<DTYPE> *pInput, int inputsize, int hiddensize, int outputsize, int use_bias = TRUE, std::string pName = "No Name") : Module<DTYPE>(pName) {
        Alloc(pInput, inputsize, hiddensize, outputsize, use_bias, pName);
    }
    virtual ~GRULayer() {}

    int Alloc(Operator<DTYPE> *pInput, int inputsize, int hiddensize, int outputsize, int use_bias, std::string pName) {
        this->SetInput(pInput);

        Operator<DTYPE> *out = pInput;

        //weight
        Tensorholder<DTYPE> *pWeightIG = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, 2*hiddensize, inputsize, 0.0, 0.01), "GRULayer_pWeight_IG_" + pName);
        Tensorholder<DTYPE> *pWeightHG = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, 2*hiddensize, hiddensize, 0.0, 0.01), "GRULayer_pWeight_HG_" + pName);
        Tensorholder<DTYPE> *pWeightICH = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, inputsize, 0.0, 0.01), "GRULayer_pWeight_HG_" + pName);
        Tensorholder<DTYPE> *pWeightHCH = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, hiddensize, 0.0, 0.01), "GRULayer_pWeight_HG_" + pName);

        //Hidden to output weight
        Tensorholder<DTYPE> *pWeight_h2o = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, outputsize, hiddensize, 0.0, 0.01), "GRULayer_pWeight_HO_" + pName);

        //bias
        Tensorholder<DTYPE> *gBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, 2*hiddensize, 0.f), "RNN_Bias_f" + pName);
        Tensorholder<DTYPE> *chBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, hiddensize, 0.f), "RNN_Bias_f" + pName);

        out = new GRU<DTYPE>(out, pWeightIG, pWeightHG, pWeightICH, pWeightHCH, gBias, chBias);

        out = new MatMul<DTYPE>(pWeight_h2o, out, "rnn_matmul_ho");

        if (use_bias) {
            Tensorholder<DTYPE> *pBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, outputsize, 0.f), "Add_Bias_" + pName);
            out = new AddColWise<DTYPE>(out, pBias, "Layer_Add_" + pName);
        }

        this->AnalyzeGraph(out);

        return TRUE;
    }
};

#endif
