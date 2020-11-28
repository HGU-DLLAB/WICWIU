#ifndef GRU_H_
#define GRU_H_    value

#include "../Operator.hpp"
#include "MatMul.hpp"
#include "Add.hpp"
#include "Tanh.hpp"
#include "Tensorholder.hpp"

#define TIME     0
#define BATCH    1

template<typename DTYPE> class GRU : public Operator<DTYPE>{
private:

    Operator<DTYPE> *MatMul_I2G;
    Operator<DTYPE> *MatMul_H2RZ;
    Operator<DTYPE> *AddGates;
    Operator<DTYPE> *AddgBias;

    //Reset Gate
    Operator<DTYPE> *RGateInput;
    Operator<DTYPE> *RGateSigmoid;

    //Update Gate(z)
    Operator<DTYPE> *ZGateInput;
    Operator<DTYPE> *ZGateSigmoid;

    //Candidate Hidden
    Operator<DTYPE> *MatMul_I2CH;     //matmul
    Operator<DTYPE> *RAndHidden;      //hadamard
    Operator<DTYPE> *MatMul_H2CH;     //matmul
    Operator<DTYPE> *BeforeCandidateHiddenInput;
    Operator<DTYPE> *CandidateHiddenInput;            //bias
    Operator<DTYPE> *CandidateHiddenTanh;

    //Hidden state
    Operator<DTYPE> *BeforeZHidden;
    Operator<DTYPE> *BeforeGHidden1;
    Operator<DTYPE> *BeforeGHidden2;
    Operator<DTYPE> *Hidden;

    //Onettme
    Operator<DTYPE> *m_aOneTensor;

    //time
    Operator<DTYPE> *m_aTempHidden;


public:
  GRU(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightIG, Operator<DTYPE> *pWeightHG, Operator<DTYPE> *pWeightICH, Operator<DTYPE> *pWeightHCH, Operator<DTYPE> *gBias, Operator<DTYPE> *chBias)
       : Operator<DTYPE>(7, pInput, pWeightIG, pWeightHG, pWeightICH, pWeightHCH, gBias, chBias) {
      #if __DEBUG__
      std::cout << "GRU::GRU(Operator<DTYPE> *)" << '\n';
      #endif  // __DEBUG__
      this->Alloc(pInput, pWeightIG, pWeightHG, pWeightICH, pWeightHCH, gBias, chBias);
  }

    GRU(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightIG, Operator<DTYPE> *pWeightHG, Operator<DTYPE> *pWeightICH, Operator<DTYPE> *pWeightHCH, Operator<DTYPE> *gBias, Operator<DTYPE> *chBias, std::string pName)
         : Operator<DTYPE>(7, pInput, pWeightIG, pWeightHG, pWeightICH, pWeightHCH, gBias, chBias) {
        #if __DEBUG__
        std::cout << "GRU::GRU(Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, pWeightIG, pWeightHG, pWeightICH, pWeightHCH, gBias, chBias);
    }

    ~GRU() {
        #if __DEBUG__
        std::cout << "GRU::~GRU()" << '\n';
        #endif  // __DEBUG__

        Delete();
    }

    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightIG, Operator<DTYPE> *pWeightHG, Operator<DTYPE> *pWeightICH, Operator<DTYPE> *pWeightHCH, Operator<DTYPE> *gBias, Operator<DTYPE> *chBias) {

        Shape *InputShape    = pInput->GetResult()->GetShape();
        Shape *WeightIGShape = pWeightIG->GetResult()->GetShape();

        int hidTimeSize  = (*InputShape)[TIME];
        int hidBatchSize = (*InputShape)[BATCH];
        int hidColSize   = (*WeightIGShape)[3]/2;

        //Onetensor
        m_aOneTensor    = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(hidTimeSize, hidBatchSize, 1, 1, hidColSize, 1.0), "tempHidden");

        //time
        m_aTempHidden   = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(hidTimeSize, hidBatchSize, 1, 1, hidColSize), "tempHidden");

        //Reset&update gate matmul, bias
        MatMul_I2G    = new MatMul<DTYPE>(pWeightIG, pInput, "gru_matmul_IG");
        MatMul_H2RZ   = new MatMul<DTYPE>(pWeightHG, m_aTempHidden, "gru_matmul_IG");
        AddGates      = new Addall<DTYPE>(MatMul_I2G, MatMul_H2RZ, "gru_addall");
        AddgBias      = new AddColWise<DTYPE>(AddGates, gBias, "gru_F_addall");

        //R Gate
        RGateInput    = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(hidTimeSize, hidBatchSize, 1, 1, hidColSize), "gru_R_addall");
        RGateSigmoid  = new Sigmoid<DTYPE>(RGateInput, "gru_R_sigmoid");

        //Z Gate
        ZGateInput    = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(hidTimeSize, hidBatchSize, 1, 1, hidColSize), "gru_Z_addall");
        ZGateSigmoid  = new Sigmoid<DTYPE>(ZGateInput, "gru_Z_sigmoid");

        //Candidate Hidden
        MatMul_I2CH   = new MatMul<DTYPE>(pWeightICH, pInput, "gru_matmul_IG");
        RAndHidden    = new Hadamard<DTYPE>(RGateSigmoid, m_aTempHidden, "ForgetGateCell");
        MatMul_H2CH   = new MatMul<DTYPE>(pWeightHCH, RAndHidden, "gru_matmul_IG");
        BeforeCandidateHiddenInput  = new Addall<DTYPE>(MatMul_I2CH, MatMul_H2CH, "gru_addall");       //2개 합치고
        CandidateHiddenInput        = new AddColWise<DTYPE>(BeforeCandidateHiddenInput, chBias, "gru_F_addall");            //bias
        CandidateHiddenTanh         = new Tanh<DTYPE>(CandidateHiddenInput, "lstm_c_tanh");

        //Hidden state
        BeforeZHidden  = new Hadamard<DTYPE>(ZGateSigmoid, m_aTempHidden, "ForgetGateCell");
        BeforeGHidden1 = new Minus<DTYPE>(m_aOneTensor, ZGateSigmoid, "new data");
        BeforeGHidden2 = new Hadamard<DTYPE>(BeforeGHidden1, CandidateHiddenTanh, "ForgetGateCell");
        Hidden         = new Addall<DTYPE>(BeforeZHidden, BeforeGHidden2, "gru_addall");

        //For AnalyzeGraph
        pInput->GetOutputContainer()->Pop(MatMul_I2G);
        pInput->GetOutputContainer()->Pop(MatMul_I2CH);
        pWeightIG->GetOutputContainer()->Pop(MatMul_I2G);
        pWeightHG->GetOutputContainer()->Pop(MatMul_H2RZ);
        pWeightICH->GetOutputContainer()->Pop(MatMul_I2CH);
        pWeightHCH->GetOutputContainer()->Pop(MatMul_H2CH);
        gBias->GetOutputContainer()->Pop(AddgBias);
        chBias->GetOutputContainer()->Pop(CandidateHiddenInput);

        Shape *ResultShape = Hidden->GetResult()->GetShape();

        int timeSize  = (*ResultShape)[TIME];
        int batchSize = (*ResultShape)[BATCH];
        int colSize   = (*ResultShape)[4];

        this->SetResult(Tensor<DTYPE>::Zeros(timeSize, batchSize, 1, 1, colSize));
        this->SetGradient(Tensor<DTYPE>::Zeros(timeSize, batchSize, 1, 1, colSize));

        return TRUE;
    }

    void Delete() {}


    int  ForwardPropagate(int pTime = 0) {

        if(pTime != 0){

            //hidden
            Tensor<DTYPE> *prevHidden = Hidden->GetResult();
            Tensor<DTYPE> *tempHidden = m_aTempHidden->GetResult();

            int batchsize      = prevHidden->GetBatchSize();
            int colSize        = prevHidden->GetColSize();
            Shape *HiddenShape = prevHidden->GetShape();

            for(int ba=0; ba<batchsize; ba++){
                for (int i = 0; i < colSize; i++) {
                    (*tempHidden)[Index5D(HiddenShape, pTime, ba, 0, 0, i)] = (*prevHidden)[Index5D(HiddenShape, pTime - 1, ba, 0, 0, i)];
                }
            }
        }

        MatMul_I2G->ForwardPropagate(pTime);
        MatMul_H2RZ->ForwardPropagate(pTime);
        AddGates->ForwardPropagate(pTime);
        AddgBias->ForwardPropagate(pTime);


        Tensor<DTYPE> *tempRGates  = RGateInput->GetResult();
        Tensor<DTYPE> *tempZGates   = ZGateInput->GetResult();

        Shape *EachShape = tempRGates->GetShape();
        Tensor<DTYPE> *OneGates = AddgBias->GetResult();

        int batchsize = OneGates->GetBatchSize();
        int h = Hidden->GetResult()->GetColSize();
        Shape *OneShape   = OneGates->GetShape();

        for(int ba=0; ba<batchsize; ba++){
            for(int i=0; i<h; i++){
                (*tempRGates)[Index5D(EachShape, pTime, ba, 0, 0, i)]   = (*OneGates)[Index5D(OneShape, pTime, ba, 0, 0, i)];
                (*tempZGates)[Index5D(EachShape, pTime, ba, 0, 0, i)]   = (*OneGates)[Index5D(OneShape, pTime, ba, 0, 0, h+i)];
            }
        }

        //R Gate
        RGateSigmoid->ForwardPropagate(pTime);

        //Z Gate
        ZGateSigmoid->ForwardPropagate(pTime);

        //Candidate Hidden
        MatMul_I2CH->ForwardPropagate(pTime);
        RAndHidden->ForwardPropagate(pTime);
        MatMul_H2CH->ForwardPropagate(pTime);
        BeforeCandidateHiddenInput->ForwardPropagate(pTime);
        CandidateHiddenInput->ForwardPropagate(pTime);
        CandidateHiddenTanh->ForwardPropagate(pTime);

        //Hidden state
        BeforeZHidden->ForwardPropagate(pTime);
        BeforeGHidden1->ForwardPropagate(pTime);
        BeforeGHidden2->ForwardPropagate(pTime);
        Hidden->ForwardPropagate(pTime);

        Tensor<DTYPE> *_result = Hidden->GetResult();
        Tensor<DTYPE> *result  = this->GetResult();

        int colSize        = result->GetColSize();
        Shape *ResultShape = result->GetShape();

        for(int ba=0; ba<batchsize; ba++){
            for (int i = 0; i < colSize; i++) {
                (*result)[Index5D(ResultShape, pTime, ba, 0, 0, i)] = (*_result)[Index5D(ResultShape, pTime, ba, 0, 0, i)];
            }
        }

        return TRUE;
    }


    int BackPropagate(int pTime = 0) {

        Tensor<DTYPE> *_grad = Hidden->GetGradient();
        Tensor<DTYPE> *grad  = this->GetGradient();

        int colSize        = grad->GetColSize();
        int timeSize       = grad->GetTimeSize();
        int batchsize      = grad->GetBatchSize();
        Shape *ResultShape = grad->GetShape();

        for(int ba=0; ba<batchsize; ba++){
            for (int i = 0; i < colSize; i++) {
                (*_grad)[Index5D(ResultShape, pTime, ba, 0, 0, i)] = (*grad)[Index5D(ResultShape, pTime, ba, 0, 0, i)];
            }
        }

        if (pTime != timeSize-1) {

            Tensor<DTYPE> *tempHiddenGrad = m_aTempHidden->GetGradient();
            Tensor<DTYPE> *prevHiddenGrad = Hidden->GetGradient();

            int colSize        = tempHiddenGrad->GetColSize();
            Shape *HiddenShape = tempHiddenGrad->GetShape();

            for(int ba=0; ba<batchsize; ba++){
                for (int i = 0; i < colSize; i++) {
                    (*prevHiddenGrad)[Index5D(HiddenShape, pTime, ba, 0, 0, i)] += (*tempHiddenGrad)[Index5D(HiddenShape, pTime+1, ba, 0, 0, i)];
                }
            }
        }

        //Hidden state
        Hidden->BackPropagate(pTime);
        BeforeGHidden2->BackPropagate(pTime);
        BeforeGHidden1->BackPropagate(pTime);
        BeforeZHidden->BackPropagate(pTime);

        //Candidate Hidden
        CandidateHiddenTanh->BackPropagate(pTime);
        CandidateHiddenInput->BackPropagate(pTime);
        BeforeCandidateHiddenInput->BackPropagate(pTime);
        MatMul_H2CH->BackPropagate(pTime);
        RAndHidden->BackPropagate(pTime);
        MatMul_I2CH->BackPropagate(pTime);

        //Z Gate
        ZGateSigmoid->BackPropagate(pTime);

        //R Gate
        RGateSigmoid->BackPropagate(pTime);


        //Gradient
        Tensor<DTYPE> *tempRGates  = RGateInput->GetGradient();
        Tensor<DTYPE> *tempZGates   = ZGateInput->GetGradient();

        Shape *EachShape = tempZGates->GetShape();

        Tensor<DTYPE> *OneGates = AddgBias->GetGradient();
        Shape *OneShape   = OneGates->GetShape();

        int h = Hidden->GetResult()->GetColSize();

        for(int ba=0; ba<batchsize; ba++){
            for(int i=0; i<h; i++){
                (*OneGates)[Index5D(OneShape, pTime, ba, 0, 0, i)]    = (*tempRGates)[Index5D(EachShape, pTime, ba, 0, 0, i)];
                (*OneGates)[Index5D(OneShape, pTime, ba, 0, 0, h+i)]   = (*tempZGates)[Index5D(EachShape, pTime, ba, 0, 0, i)];
            }
        }

        AddgBias->BackPropagate(pTime);
        AddGates->BackPropagate(pTime);
        MatMul_H2RZ->BackPropagate(pTime);
        MatMul_I2G->BackPropagate(pTime);

        return TRUE;
    }


    int ResetResult() {

        //matmul, bias
        MatMul_I2G->ResetResult();
        MatMul_H2RZ->ResetResult();
        AddGates->ResetResult();
        AddgBias->ResetResult();

        //R Gate
        RGateInput->ResetResult();
        RGateSigmoid->ResetResult();

        //Z Gate
        ZGateInput->ResetResult();
        ZGateSigmoid->ResetResult();

        //Candidate Hidden
        MatMul_I2CH->ResetResult();
        RAndHidden->ResetResult();
        MatMul_H2CH->ResetResult();
        BeforeCandidateHiddenInput->ResetResult();
        CandidateHiddenInput->ResetResult();
        CandidateHiddenTanh->ResetResult();

        //Hidden state
        BeforeZHidden->ResetResult();
        BeforeGHidden1->ResetResult();
        BeforeGHidden2->ResetResult();
        Hidden->ResetResult();

        m_aTempHidden->ResetResult();

        Tensor<DTYPE> *result = this->GetResult();
        result->Reset();

    }

    int ResetGradient() {

        //matmul, bias
        MatMul_I2G->ResetGradient();
        MatMul_H2RZ->ResetGradient();
        AddGates->ResetGradient();
        AddgBias->ResetGradient();

        //R Gate
        RGateInput->ResetGradient();
        RGateSigmoid->ResetGradient();

        //Z Gate
        ZGateInput->ResetGradient();
        ZGateSigmoid->ResetGradient();

        //Candidate Hidden
        MatMul_I2CH->ResetGradient();
        RAndHidden->ResetGradient();
        MatMul_H2CH->ResetGradient();
        BeforeCandidateHiddenInput->ResetGradient();
        CandidateHiddenInput->ResetGradient();
        CandidateHiddenTanh->ResetGradient();

        //Hidden state
        BeforeZHidden->ResetGradient();
        BeforeGHidden1->ResetGradient();
        BeforeGHidden2->ResetGradient();
        Hidden->ResetGradient();

        //time 처리
        m_aTempHidden->ResetGradient();

        Tensor<DTYPE> *grad = this->GetGradient();
        grad->Reset();

    }


};


#endif  // GRU_H_
