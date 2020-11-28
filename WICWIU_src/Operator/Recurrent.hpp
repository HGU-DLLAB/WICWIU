#ifndef RECURRENT_H_
#define RECURRENT_H_    value

#include "../Operator.hpp"
#include "MatMul.hpp"
#include "Add.hpp"
#include "Tanh.hpp"
#include "Tensorholder.hpp"

#define TIME     0
#define BATCH    1

template<typename DTYPE> class Recurrent : public Operator<DTYPE>{
private:
    Operator<DTYPE> *m_aInput2Hidden;
    Operator<DTYPE> *m_aHidden2Hidden;
    Operator<DTYPE> *m_aTempHidden;
    Operator<DTYPE> *m_aPrevActivate;
    Operator<DTYPE> *ApplyActivation;
    Operator<DTYPE> *AddBias;

#ifdef __CUDNN__
    //cudnnRNNDescriptor_t rnnDesc;
    //  cudnnRNNDataDescriptor_t RNNDataDesc;
    DTYPE m_alpha;
    ///< 연산 간 두 Operand의 가중치를 표현하기 위한 변수. ex) z = α*x + β*y
    DTYPE m_beta;
    ///< 연산 간 두 Operand의 가중치를 표현하기 위한 변수. ex) z = α*x + β*y
#endif  // __CUDNN__

public:
    Recurrent(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightIH, Operator<DTYPE> *pWeightHH, Operator<DTYPE> *rBias) : Operator<DTYPE>(4, pInput, pWeightIH, pWeightHH, rBias) {
        #if __DEBUG__
        std::cout << "Recurrent::Recurrent(Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, pWeightIH, pWeightHH, rBias);
    }

    Recurrent(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightIH, Operator<DTYPE> *pWeightHH, Operator<DTYPE> *rBias, std::string pName) : Operator<DTYPE>(4, pInput, pWeightIH, pWeightHH, rBias, pName) {
        #if __DEBUG__
        std::cout << "Recurrent::Recurrent(Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, pWeightIH, pWeightHH, rBias);
    }

    ~Recurrent() {
        #if __DEBUG__
        std::cout << "Recurrent::~Recurrent()" << '\n';
        #endif  // __DEBUG__

        Delete();
    }

    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightIH, Operator<DTYPE> *pWeightHH, Operator<DTYPE> *rBias) {

        Shape *InputShape    = pInput->GetResult()->GetShape();
        Shape *WeightXHShape = pWeightIH->GetResult()->GetShape();

        int hidTimeSize  = (*InputShape)[TIME];
        int hidBatchSize = (*InputShape)[BATCH];
        int hidColSize   = (*WeightXHShape)[3];

        m_aInput2Hidden  = new MatMul<DTYPE>(pWeightIH, pInput, "rnn_matmul_xh");
        m_aTempHidden    = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(hidTimeSize, hidBatchSize, 1, 1, hidColSize), "tempHidden");
        m_aHidden2Hidden = new MatMul<DTYPE>(pWeightHH, m_aTempHidden, "rnn_matmul_hh");
        m_aPrevActivate  = new Addall<DTYPE>(m_aInput2Hidden, m_aHidden2Hidden, "rnn_addall");
        AddBias = new AddColWise<DTYPE>(m_aPrevActivate, rBias, "net_with_bias_");
        ApplyActivation  = new Tanh<DTYPE>(AddBias, "rnn_tanh");

        //ApplyActivation  = new Relu<DTYPE>(AddBias, "rnn_tanh");

        //For AnalyzeGraph
        pInput->GetOutputContainer()->Pop(m_aInput2Hidden);
        rBias->GetOutputContainer()->Pop(AddBias);
        pWeightIH->GetOutputContainer()->Pop(m_aInput2Hidden);
        pWeightHH->GetOutputContainer()->Pop(m_aHidden2Hidden);

        Shape *ResultShape = ApplyActivation->GetResult()->GetShape();

        int timeSize  = (*ResultShape)[TIME];
        int batchSize = (*ResultShape)[BATCH];
        int colSize   = (*ResultShape)[4];

        this->SetResult(Tensor<DTYPE>::Zeros(timeSize, batchSize, 1, 1, colSize));
        this->SetGradient(Tensor<DTYPE>::Zeros(timeSize, batchSize, 1, 1, colSize));

        return TRUE;
    }

#if __CUDNN__
      void InitializeAttributeForGPU(unsigned int idOfDevice) {

          m_alpha = 1;
          m_beta  = 0;

          m_aInput2Hidden->SetDeviceGPU(this->GetCudnnHandle(), idOfDevice);

          m_aTempHidden->SetDeviceGPU(this->GetCudnnHandle(), idOfDevice);

          m_aHidden2Hidden->SetDeviceGPU(this->GetCudnnHandle(), idOfDevice);

          m_aPrevActivate->SetDeviceGPU(this->GetCudnnHandle(), idOfDevice);

          AddBias->SetDeviceGPU(this->GetCudnnHandle(), idOfDevice);

          ApplyActivation->SetDeviceGPU(this->GetCudnnHandle(), idOfDevice);

      }

#endif  // if __CUDNN__


    void Delete() {}


    int  ForwardPropagate(int pTime = 0) {

        m_aInput2Hidden->ForwardPropagate(pTime);

        if (pTime != 0) {
            Tensor<DTYPE> *prevHidden = ApplyActivation->GetResult();
            Tensor<DTYPE> *tempHidden = m_aTempHidden->GetResult();

            int colSize        = prevHidden->GetColSize();
            Shape *HiddenShape = prevHidden->GetShape();

            for (int i = 0; i < colSize; i++) {
                (*tempHidden)[Index5D(HiddenShape, pTime, 0, 0, 0, i)] = (*prevHidden)[Index5D(HiddenShape, pTime - 1, 0, 0, 0, i)];
            }

            m_aHidden2Hidden->ForwardPropagate(pTime);
        }
        m_aPrevActivate->ForwardPropagate(pTime);

        AddBias->ForwardPropagate(pTime);

        ApplyActivation->ForwardPropagate(pTime);

        Tensor<DTYPE> *_result = ApplyActivation->GetResult();
        Tensor<DTYPE> *result  = this->GetResult();

        int colSize        = result->GetColSize();
        Shape *ResultShape = result->GetShape();

        for (int i = 0; i < colSize; i++) {
            (*result)[Index5D(ResultShape, pTime, 0, 0, 0, i)] = (*_result)[Index5D(ResultShape, pTime, 0, 0, 0, i)];
        }

        return TRUE;
    }

    int BackPropagate(int pTime = 0) {

        Tensor<DTYPE> *_grad = ApplyActivation->GetGradient();
        Tensor<DTYPE> *grad  = this->GetGradient();

        int colSize        = grad->GetColSize();
        int timeSize       = grad->GetTimeSize();
        Shape *ResultShape = grad->GetShape();

        for (int i = 0; i < colSize; i++) {
            (*_grad)[Index5D(ResultShape, pTime, 0, 0, 0, i)] = (*grad)[Index5D(ResultShape, pTime, 0, 0, 0, i)];
        }

        if (pTime != timeSize-1) {
            m_aHidden2Hidden->BackPropagate(pTime+1);

            Tensor<DTYPE> *tempHiddenGrad = m_aTempHidden->GetGradient();
            Tensor<DTYPE> *prevHiddenGrad = ApplyActivation->GetGradient();

            int colSize        = tempHiddenGrad->GetColSize();
            Shape *HiddenShape = tempHiddenGrad->GetShape();

            for (int i = 0; i < colSize; i++) {
                (*prevHiddenGrad)[Index5D(HiddenShape, pTime, 0, 0, 0, i)] += (*tempHiddenGrad)[Index5D(HiddenShape, pTime+1, 0, 0, 0, i)];
            }
        }

        ApplyActivation->BackPropagate(pTime);

        AddBias->BackPropagate(pTime);

        m_aPrevActivate->BackPropagate(pTime);

        m_aInput2Hidden->BackPropagate(pTime);

        return TRUE;
    }

    #if __CUDNN__
        int ForwardPropagateOnGPU(int pTime = 0) {

            cudnnTensorDescriptor_t desc = NULL;

            m_aInput2Hidden->ForwardPropagateOnGPU(pTime);

            if (pTime != 0) {
                Tensor<DTYPE> *prevHidden = ApplyActivation->GetResult();
                Tensor<DTYPE> *tempHidden = m_aTempHidden->GetResult();

                DTYPE *pDevPrevHidden = prevHidden->GetGPUData(pTime - 1);
                DTYPE *pDevTempHidden = tempHidden->GetGPUData(pTime);

                desc = prevHidden->GetDescriptor();

                checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
                                          &m_alpha, desc, pDevPrevHidden,
                                          &m_beta, desc, pDevTempHidden));

                m_aHidden2Hidden->ForwardPropagateOnGPU(pTime);
            }

            m_aPrevActivate->ForwardPropagateOnGPU(pTime);

            AddBias->ForwardPropagateOnGPU(pTime);

            ApplyActivation->ForwardPropagateOnGPU(pTime);

            Tensor<DTYPE> *_result = ApplyActivation->GetResult();
            Tensor<DTYPE> *result  = this->GetResult();

            DTYPE *_pDevResult = _result->GetGPUData(pTime);
            DTYPE *pDevresult = result->GetGPUData(pTime);

            desc = result->GetDescriptor();

            checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
                                      &m_alpha, desc, _pDevResult,
                                      &m_beta, desc, pDevresult));

            return TRUE;
        }


        int BackPropagateOnGPU(int pTime = 0) {

            cudnnTensorDescriptor_t desc = NULL;

            Tensor<DTYPE> *_grad = ApplyActivation->GetGradient();
            Tensor<DTYPE> *grad  = this->GetGradient();

            DTYPE *_pDevGrad = _grad->GetGPUData(pTime);
            DTYPE *pDevGrad = grad->GetGPUData(pTime);

            desc = grad->GetDescriptor();

            checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
                                      &m_alpha, desc, pDevGrad,
                                      &m_beta, desc, _pDevGrad));

            int timeSize       = grad->GetTimeSize();

            if (pTime != timeSize-1) {
                m_aHidden2Hidden->BackPropagateOnGPU(pTime+1);

                Tensor<DTYPE> *tempHiddenGrad = m_aTempHidden->GetGradient();
                Tensor<DTYPE> *prevHiddenGrad = ApplyActivation->GetGradient();

                DTYPE *pDevTempHiddenGrad = tempHiddenGrad->GetGPUData(pTime + 1);
                DTYPE *pDevPrevHiddenGrad = prevHiddenGrad->GetGPUData(pTime);

                desc = tempHiddenGrad->GetDescriptor();

                checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
                                          &m_alpha, desc, pDevTempHiddenGrad,
                                          &m_alpha, desc, pDevPrevHiddenGrad));
            }

            ApplyActivation->BackPropagateOnGPU(pTime);


            AddBias->BackPropagateOnGPU(pTime);

            m_aPrevActivate->BackPropagateOnGPU(pTime);

            m_aInput2Hidden->BackPropagateOnGPU(pTime);


            return TRUE;
        }
    #endif  // if __CUDNN__


        int ResetResult() {
            m_aInput2Hidden->ResetResult();
            m_aHidden2Hidden->ResetResult();
            m_aTempHidden->ResetResult();
            m_aPrevActivate->ResetResult();
            ApplyActivation->ResetResult();
            AddBias->ResetResult();

            Tensor<DTYPE> *result = this->GetResult();
            result->Reset();

        }

        int ResetGradient() {
            m_aInput2Hidden->ResetGradient();
            m_aHidden2Hidden->ResetGradient();
            m_aTempHidden->ResetGradient();
            m_aPrevActivate->ResetGradient();
            ApplyActivation->ResetGradient();
            AddBias->ResetGradient();

            Tensor<DTYPE> *grad = this->GetGradient();
            grad->Reset();
        }


    };


    #endif  // RECURRENT_H_
