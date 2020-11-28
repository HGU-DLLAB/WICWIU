#ifndef __LSTM_LAYER__
#define __LSTM_LAYER__    value  

#include "../Module.hpp"


template<typename DTYPE> class LSTMLayer : public Module<DTYPE>{
private:
public:
    /*!
    @brief LSTMLayer 클래스 생성자
    @details LSTMLayer 클래스의 Alloc 함수를 호출한다.*/
    LSTMLayer(Operator<DTYPE> *pInput, int inputsize, int hiddensize, int outputsize, int use_bias = TRUE, std::string pName = "No Name") : Module<DTYPE>(pName) {
        Alloc(pInput, inputsize, hiddensize, outputsize, use_bias, pName);
    }

    /*!
    @brief LSTMLayer 클래스 소멸자
    @details 단, 동적 할당 받은 Operator들은 NeuralNetwork에서 할당 해제한다.  */
    virtual ~LSTMLayer() {}

    /*!
    @brief LSTMLayer 그래프를 동적으로 할당 및 구성하는 메소드
    @details Input Operator의 Element에 대해 2D Convolution을 수행한다.
    @param pInput
    @param use_bias Bias 사용 유무, 0일 시 사용 안 함, 0이 아닐 시 사용
    @param pName Module의 이름
    @return TRUE
    @see
    */
    int Alloc(Operator<DTYPE> *pInput, int inputsize, int hiddensize, int outputsize, int use_bias, std::string pName) {
        this->SetInput(pInput);

        Operator<DTYPE> *out = pInput;


        //weight 8개
        Tensorholder<DTYPE> *pWeight_IF = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, inputsize, 0.0, 0.01), "LSTMLayer_pWeight_IF_" + pName);
        Tensorholder<DTYPE> *pWeight_HF = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, hiddensize, 0.0, 0.01), "LSTMLayer_pWeight_HF_" + pName);

        Tensorholder<DTYPE> *pWeight_II = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, inputsize, 0.0, 0.01), "LSTMLayer_pWeight_II_" + pName);
        Tensorholder<DTYPE> *pWeight_HI = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, hiddensize, 0.0, 0.01), "LSTMLayer_pWeight_HI_" + pName);

        Tensorholder<DTYPE> *pWeight_IC = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, inputsize, 0.0, 0.01), "LSTMLayer_pWeight_IC_" + pName);
        Tensorholder<DTYPE> *pWeight_HC = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, hiddensize, 0.0, 0.01), "LSTMLayer_pWeight_HC_" + pName);

        Tensorholder<DTYPE> *pWeight_IO = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, inputsize, 0.0, 0.01), "LSTMLayer_pWeight_IO_" + pName);
        Tensorholder<DTYPE> *pWeight_HO = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, hiddensize, 0.0, 0.01), "LSTMLayer_pWeight_HO_" + pName);

        //output weight
        Tensorholder<DTYPE> *pWeight_h2o = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, outputsize, hiddensize, 0.0, 0.01), "LSTMLayer_pWeight_HO_" + pName);


        //bias 4개
        Tensorholder<DTYPE> *fBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, hiddensize, 0.f), "RNN_Bias_f" + pName);
        Tensorholder<DTYPE> *iBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, hiddensize, 0.f), "RNN_Bias_i" + pName);
        Tensorholder<DTYPE> *cBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, hiddensize, 0.f), "RNN_Bias_c" + pName);
        Tensorholder<DTYPE> *oBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, hiddensize, 0.f), "RNN_Bias_o" + pName);

        out = new LSTM<DTYPE>(out, pWeight_IF, pWeight_HF, pWeight_II, pWeight_HI, pWeight_IC, pWeight_HC, pWeight_IO, pWeight_HO, fBias, iBias, cBias, oBias);

        out = new MatMul<DTYPE>(pWeight_h2o, out, "rnn_matmul_ho");



        if (use_bias) {
            Tensorholder<DTYPE> *pBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, outputsize, 0.f), "Add_Bias_" + pName);
            out = new AddColWise<DTYPE>(out, pBias, "Layer_Add_" + pName);
        }

        this->AnalyzeGraph(out);

        return TRUE;
    }
};


#endif  // __RECURRENT_LAYER__
