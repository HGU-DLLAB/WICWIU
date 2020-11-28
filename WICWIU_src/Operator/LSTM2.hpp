#ifndef LSTM2_H_
#define LSTM2_H_    value

#include "../Operator.hpp"
#include "MatMul.hpp"
#include "Add.hpp"
#include "Tanh.hpp"
#include "Tensorholder.hpp"

#define TIME     0
#define BATCH    1

template<typename DTYPE> class LSTM2 : public Operator<DTYPE>{
private:

    //전체 matmul, bias
    Operator<DTYPE> *MatMul_I2G;
    Operator<DTYPE> *MatMul_H2G;
    Operator<DTYPE> *AddGates;
    Operator<DTYPE> *AddBias;

    //forget Gate
    Operator<DTYPE> *ForgetGateInput;
    Operator<DTYPE> *ForgetGateSigmoid;

    //Input Gate
    Operator<DTYPE> *InputGateInput;
    Operator<DTYPE> *InputGateSigmoid;

    //Cell Gate
    Operator<DTYPE> *CellGateInput;
    Operator<DTYPE> *CellGateTanh;

    //Output Gate
    Operator<DTYPE> *OutputGateInput;
    Operator<DTYPE> *OutputGateSigmoid;

    //Cell state
    Operator<DTYPE> *ForgetGateCell;
    Operator<DTYPE> *InputGateCell;
    Operator<DTYPE> *CellState;

    //Hidden state
    Operator<DTYPE> *BeforeHidden;
    Operator<DTYPE> *Hidden;

    //time 처리
    Operator<DTYPE> *m_aTempHidden;
    Operator<DTYPE> *m_TempCellState;

#ifdef __CUDNN__
    //cudnnRNNDescriptor_t rnnDesc;
    //  cudnnRNNDataDescriptor_t RNNDataDesc;
    DTYPE m_alpha;
    ///< 연산 간 두 Operand의 가중치를 표현하기 위한 변수. ex) z = α*x + β*y
    DTYPE m_beta;
    ///< 연산 간 두 Operand의 가중치를 표현하기 위한 변수. ex) z = α*x + β*y
#endif  // __CUDNN__


public:
  LSTM2(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightIG, Operator<DTYPE> *pWeightHG, Operator<DTYPE> *lstmBias)
       : Operator<DTYPE>(4, pInput, pWeightIG, pWeightHG, lstmBias) {
      #if __DEBUG__
      std::cout << "LSTM::LSTM(Operator<DTYPE> *)" << '\n';
      #endif  // __DEBUG__
      this->Alloc(pInput, pWeightIG, pWeightHG, lstmBias);
  }

    LSTM2(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightIG, Operator<DTYPE> *pWeightHG, Operator<DTYPE> *lstmBias, std::string pName)
         : Operator<DTYPE>(4, pInput, pWeightIG, pWeightHG, lstmBias) {
        #if __DEBUG__
        std::cout << "LSTM::LSTM(Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, pWeightIG, pWeightHG, lstmBias);
    }

    ~LSTM2() {
        #if __DEBUG__
        std::cout << "LSTM::~LSTM()" << '\n';
        #endif  // __DEBUG__

        Delete();
    }

    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightIG, Operator<DTYPE> *pWeightHG, Operator<DTYPE> *lstmBias) {


        Shape *InputShape    = pInput->GetResult()->GetShape();
        Shape *WeightIHShape = pWeightIG->GetResult()->GetShape();

        int hidTimeSize  = (*InputShape)[TIME];
        int hidBatchSize = (*InputShape)[BATCH];
        int hidColSize   = (*WeightIHShape)[3]/4;

        std::cout<<"hidColSize = "<<hidColSize<<'\n';

        //time
        m_aTempHidden         = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(hidTimeSize, hidBatchSize, 1, 1, hidColSize), "tempHidden");
        m_TempCellState       = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(hidTimeSize, hidBatchSize, 1, 1, hidColSize), "tempCell");

        MatMul_I2G            = new MatMul<DTYPE>(pWeightIG, pInput, "lstm_matmul_IG");
        MatMul_H2G            = new MatMul<DTYPE>(pWeightHG, m_aTempHidden, "lstm_matmul_HG");
        AddGates              = new Addall<DTYPE>(MatMul_I2G, MatMul_H2G, "lstm_addall");
        AddBias               = new AddColWise<DTYPE>(AddGates, lstmBias, "lstm_F_addall");

        //forget Gate
        ForgetGateInput       = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(hidTimeSize, hidBatchSize, 1, 1, hidColSize), "lstm_F_addall");
        ForgetGateSigmoid     = new Sigmoid<DTYPE>(ForgetGateInput, "lstm_f_sigmoid");

        //Input Gate
        InputGateInput        = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(hidTimeSize, hidBatchSize, 1, 1, hidColSize), "lstm_I_addall");
        InputGateSigmoid      = new Sigmoid<DTYPE>(InputGateInput, "lstm_I_sigmoid");

        //Cell Gate
        CellGateInput         = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(hidTimeSize, hidBatchSize, 1, 1, hidColSize), "lstm_c_addall");
        CellGateTanh          = new Tanh<DTYPE>(CellGateInput, "lstm_c_tanh");


        //Output Gate
        OutputGateInput       = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(hidTimeSize, hidBatchSize, 1, 1, hidColSize), "lstm_o_addall");
        OutputGateSigmoid     = new Sigmoid<DTYPE>(OutputGateInput, "lstm_o_sigmoid");

        //Cell state
        ForgetGateCell        = new Hadamard<DTYPE>(m_TempCellState, ForgetGateSigmoid, "ForgetGateCell");
        InputGateCell         = new Hadamard<DTYPE>(CellGateTanh, InputGateSigmoid, "beforecellstate");
        CellState             = new Addall<DTYPE>(ForgetGateCell, InputGateCell, "cellState");

        //Hidden state
        BeforeHidden          = new Tanh<DTYPE>(CellState, "beforehidden");
        Hidden                = new Hadamard<DTYPE>(BeforeHidden, OutputGateSigmoid, "cellstate");

        //For AnalyzeGraph
        pInput->GetOutputContainer()->Pop(MatMul_I2G);
        pWeightIG->GetOutputContainer()->Pop(MatMul_I2G);
        pWeightHG->GetOutputContainer()->Pop(MatMul_H2G);
        lstmBias->GetOutputContainer()->Pop(AddBias);

        Shape *ResultShape = Hidden->GetResult()->GetShape();

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

          MatMul_I2G->SetDeviceGPU(this->GetCudnnHandle(), idOfDevice);
          MatMul_H2G->SetDeviceGPU(this->GetCudnnHandle(), idOfDevice);
          AddGates->SetDeviceGPU(this->GetCudnnHandle(), idOfDevice);
          AddBias->SetDeviceGPU(this->GetCudnnHandle(), idOfDevice);

          //forget Gate
          ForgetGateInput->SetDeviceGPU(this->GetCudnnHandle(), idOfDevice);
          ForgetGateSigmoid->SetDeviceGPU(this->GetCudnnHandle(), idOfDevice);

          //Input Gate
          InputGateInput->SetDeviceGPU(this->GetCudnnHandle(), idOfDevice);
          InputGateSigmoid->SetDeviceGPU(this->GetCudnnHandle(), idOfDevice);

          //???
          CellGateInput->SetDeviceGPU(this->GetCudnnHandle(), idOfDevice);
          CellGateTanh->SetDeviceGPU(this->GetCudnnHandle(), idOfDevice);

          //Output Gate
          OutputGateInput->SetDeviceGPU(this->GetCudnnHandle(), idOfDevice);
          OutputGateSigmoid->SetDeviceGPU(this->GetCudnnHandle(), idOfDevice);

          //Cell state
          ForgetGateCell->SetDeviceGPU(this->GetCudnnHandle(), idOfDevice);
          InputGateCell->SetDeviceGPU(this->GetCudnnHandle(), idOfDevice);
          CellState->SetDeviceGPU(this->GetCudnnHandle(), idOfDevice);

          //Hidden state
          BeforeHidden->SetDeviceGPU(this->GetCudnnHandle(), idOfDevice);
          Hidden->SetDeviceGPU(this->GetCudnnHandle(), idOfDevice);

          //time
          m_aTempHidden->SetDeviceGPU(this->GetCudnnHandle(), idOfDevice);
          m_TempCellState->SetDeviceGPU(this->GetCudnnHandle(), idOfDevice);

      }

#endif  // if __CUDNN__

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

            //Cell
            Tensor<DTYPE> *prevCellState = CellState->GetResult();
            Tensor<DTYPE> *tempCellState = m_TempCellState->GetResult();

            colSize        = prevCellState->GetColSize();
            Shape *CellShape = prevCellState->GetShape();

            for(int ba=0; ba<batchsize; ba++){
                for (int i = 0; i < colSize; i++) {
                    (*tempCellState)[Index5D(CellShape, pTime, ba, 0, 0, i)] = (*prevCellState)[Index5D(CellShape, pTime - 1, ba, 0, 0, i)];
                }
            }

        }

        MatMul_I2G->ForwardPropagate(pTime);
        MatMul_H2G->ForwardPropagate(pTime);
        AddGates->ForwardPropagate(pTime);
        AddBias->ForwardPropagate(pTime);

        Tensor<DTYPE> *tempForgetGates  = ForgetGateInput->GetResult();
        Tensor<DTYPE> *tempInputGates   = InputGateInput->GetResult();
        Tensor<DTYPE> *tempCellGates    = CellGateInput->GetResult();
        Tensor<DTYPE> *tempOutputGates  = OutputGateInput->GetResult();

        Shape *EachShape = tempCellGates->GetShape();

        Tensor<DTYPE> *OneGates = AddBias->GetResult();

        int batchsize      = OneGates->GetBatchSize();
        int h = Hidden->GetResult()->GetColSize();
        Shape *OneShape   = OneGates->GetShape();

        for(int ba=0; ba<batchsize; ba++){
            for(int i=0; i<h; i++){
                (*tempForgetGates)[Index5D(EachShape, pTime, ba, 0, 0, i)]    = (*OneGates)[Index5D(OneShape, pTime, ba, 0, 0, i)];
                (*tempInputGates)[Index5D(EachShape, pTime, ba, 0, 0, i)]   = (*OneGates)[Index5D(OneShape, pTime, ba, 0, 0, h+i)];
                (*tempCellGates)[Index5D(EachShape, pTime, ba, 0, 0, i)]   = (*OneGates)[Index5D(OneShape, pTime, ba, 0, 0, 2*h+i)];
                (*tempOutputGates)[Index5D(EachShape, pTime, ba, 0, 0, i)] = (*OneGates)[Index5D(OneShape, pTime, ba, 0, 0, 3*h+i)];
            }
        }

        //Forget Gate
        ForgetGateInput->ForwardPropagate(pTime);
        ForgetGateSigmoid->ForwardPropagate(pTime);

        //Input Gate
        InputGateInput->ForwardPropagate(pTime);
        InputGateSigmoid->ForwardPropagate(pTime);

        //Cell Gate
        CellGateInput->ForwardPropagate(pTime);
        CellGateTanh->ForwardPropagate(pTime);

        //Output Gate
        OutputGateInput->ForwardPropagate(pTime);
        OutputGateSigmoid->ForwardPropagate(pTime);

        //Cell state
        ForgetGateCell->ForwardPropagate(pTime);
        InputGateCell->ForwardPropagate(pTime);
        CellState->ForwardPropagate(pTime);

        //Hidden state
        BeforeHidden->ForwardPropagate(pTime);
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

        int batchsize        = grad->GetBatchSize();
        int colSize        = grad->GetColSize();
        int timeSize       = grad->GetTimeSize();
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
        BeforeHidden->BackPropagate(pTime);

        if (pTime != timeSize-1) {

            Tensor<DTYPE> *tempCellGrad = m_TempCellState->GetGradient();
            Tensor<DTYPE> *prevCellGrad = CellState->GetGradient();

            int colSize        = tempCellGrad->GetColSize();
            Shape *CellShape = tempCellGrad->GetShape();

            for(int ba=0; ba<batchsize; ba++){
                for (int i = 0; i < colSize; i++) {
                    (*prevCellGrad)[Index5D(CellShape, pTime, ba, 0, 0, i)] += (*tempCellGrad)[Index5D(CellShape, pTime+1, ba, 0, 0, i)];
                }
            }
        }

        //Cell state
        CellState->BackPropagate(pTime);
        InputGateCell->BackPropagate(pTime);
        ForgetGateCell->BackPropagate(pTime);

        //Output Gate
        OutputGateSigmoid->BackPropagate(pTime);
        OutputGateInput->BackPropagate(pTime);

        //Cell gate
        CellGateTanh->BackPropagate(pTime);
        CellGateInput->BackPropagate(pTime);

        //Input Gates
        InputGateSigmoid->BackPropagate(pTime);
        InputGateInput->BackPropagate(pTime);

        //Forget Gates
        ForgetGateSigmoid->BackPropagate(pTime);
        ForgetGateInput->BackPropagate(pTime);

        //Gradient
        Tensor<DTYPE> *tempForgetGates  = ForgetGateInput->GetGradient();
        Tensor<DTYPE> *tempInputGates   = InputGateInput->GetGradient();
        Tensor<DTYPE> *tempCellGates    = CellGateInput->GetGradient();
        Tensor<DTYPE> *tempOutputGates  = OutputGateInput->GetGradient();
        Shape *EachShape = tempCellGates->GetShape();

        Tensor<DTYPE> *OneGates = AddBias->GetGradient();
        Shape *OneShape   = OneGates->GetShape();

        int h = Hidden->GetResult()->GetColSize();

        for(int ba=0; ba<batchsize; ba++){
            for(int i=0; i<h; i++){
                (*OneGates)[Index5D(OneShape, pTime, ba, 0, 0, i)]    = (*tempForgetGates)[Index5D(EachShape, pTime, ba, 0, 0, i)];
                (*OneGates)[Index5D(OneShape, pTime, ba, 0, 0, h+i)]   = (*tempInputGates)[Index5D(EachShape, pTime, ba, 0, 0, i)];
                (*OneGates)[Index5D(OneShape, pTime, ba, 0, 0, 2*h+i)]   = (*tempCellGates)[Index5D(EachShape, pTime, ba, 0, 0, i)];
                (*OneGates)[Index5D(OneShape, pTime, ba, 0, 0, 3*h+i)] = (*tempOutputGates)[Index5D(EachShape, pTime, ba, 0, 0, i)];
            }
        }

        AddBias->BackPropagate(pTime);
        AddGates->BackPropagate(pTime);
        MatMul_H2G->BackPropagate(pTime);
        MatMul_I2G->BackPropagate(pTime);

        return TRUE;
    }

#if __CUDNN__
    int ForwardPropagateOnGPU(int pTime = 0) {

        cudnnTensorDescriptor_t desc = NULL;

        if(pTime != 0){
            //hidden
            Tensor<DTYPE> *prevHidden = Hidden->GetResult();
            Tensor<DTYPE> *tempHidden = m_aTempHidden->GetResult();

            int colSize        = prevHidden->GetColSize();
            Shape *HiddenShape = prevHidden->GetShape();

            for (int i = 0; i < colSize; i++) {
                (*tempHidden)[Index5D(HiddenShape, pTime, 0, 0, 0, i)] = (*prevHidden)[Index5D(HiddenShape, pTime - 1, 0, 0, 0, i)];
            }

            Tensor<DTYPE> *prevCellState = CellState->GetResult();
            Tensor<DTYPE> *tempCellState = m_TempCellState->GetResult();

            colSize        = prevCellState->GetColSize();
            Shape *CellShape = prevCellState->GetShape();

            for (int i = 0; i < colSize; i++) {
                (*tempCellState)[Index5D(CellShape, pTime, 0, 0, 0, i)] = (*prevCellState)[Index5D(CellShape, pTime - 1, 0, 0, 0, i)];
            }

        }


        MatMul_I2G->ForwardPropagateOnGPU(pTime);
        MatMul_H2G->ForwardPropagateOnGPU(pTime);
        AddGates->ForwardPropagateOnGPU(pTime);
        AddBias->ForwardPropagateOnGPU(pTime);


        Tensor<DTYPE> *tempForgetGates  = ForgetGateInput->GetResult();
        Tensor<DTYPE> *tempInputGates   = InputGateInput->GetResult();
        Tensor<DTYPE> *tempCellGates    = CellGateInput->GetResult();
        Tensor<DTYPE> *tempOutputGates  = OutputGateInput->GetResult();

        Shape *EachShape = tempCellGates->GetShape();

        Tensor<DTYPE> *OneGates = AddBias->GetResult();

        int h = Hidden->GetResult()->GetColSize();
        Shape *OneShape   = OneGates->GetShape();

        for(int i=0; i<h; i++){
            (*tempForgetGates)[Index5D(EachShape, pTime, 0, 0, 0, i)]    = (*OneGates)[Index5D(OneShape, pTime, 0, 0, 0, i)];
            (*tempInputGates)[Index5D(EachShape, pTime, 0, 0, 0, i)]   = (*OneGates)[Index5D(OneShape, pTime, 0, 0, 0, h+i)];
            (*tempCellGates)[Index5D(EachShape, pTime, 0, 0, 0, i)]   = (*OneGates)[Index5D(OneShape, pTime, 0, 0, 0, 2*h+i)];
            (*tempOutputGates)[Index5D(EachShape, pTime, 0, 0, 0, i)] = (*OneGates)[Index5D(OneShape, pTime, 0, 0, 0, 3*h+i)];
        }

        //Forget Gates
        ForgetGateInput->ForwardPropagateOnGPU(pTime);
        ForgetGateSigmoid->ForwardPropagateOnGPU(pTime);

        //Input Gates
        InputGateInput->ForwardPropagateOnGPU(pTime);
        InputGateSigmoid->ForwardPropagateOnGPU(pTime);

        //???
        CellGateInput->ForwardPropagateOnGPU(pTime);
        CellGateTanh->ForwardPropagateOnGPU(pTime);

        //Output Gate
        OutputGateInput->ForwardPropagateOnGPU(pTime);
        OutputGateSigmoid->ForwardPropagateOnGPU(pTime);

        //Cell state
        ForgetGateCell->ForwardPropagateOnGPU(pTime);
        InputGateCell->ForwardPropagateOnGPU(pTime);
        CellState->ForwardPropagateOnGPU(pTime);

        //Hidden state
        BeforeHidden->ForwardPropagateOnGPU(pTime);
        Hidden->ForwardPropagateOnGPU(pTime);


        Tensor<DTYPE> *_result = Hidden->GetResult();
        Tensor<DTYPE> *result  = this->GetResult();

        int colSize        = result->GetColSize();
        Shape *ResultShape = result->GetShape();

        for (int i = 0; i < colSize; i++) {
            (*result)[Index5D(ResultShape, pTime, 0, 0, 0, i)] = (*_result)[Index5D(ResultShape, pTime, 0, 0, 0, i)];
        }

        return TRUE;

    }



    int BackPropagateOnGPU(int pTime = 0) {
    }
#endif  // if __CUDNN__


    int ResetResult() {
        //time
        m_aTempHidden->ResetResult();
        m_TempCellState->ResetResult();

        MatMul_I2G->ResetResult();
        MatMul_H2G->ResetResult();
        AddGates->ResetResult();
        AddBias->ResetResult();

        //forget Gate
        ForgetGateInput->ResetResult();
        ForgetGateSigmoid->ResetResult();

        //Input Gate
        InputGateInput->ResetResult();
        InputGateSigmoid->ResetResult();

        //???
        CellGateInput->ResetResult();
        CellGateTanh->ResetResult();

        //Output Gate
        OutputGateInput->ResetResult();
        OutputGateSigmoid->ResetResult();

        //Cell state
        ForgetGateCell->ResetResult();
        InputGateCell->ResetResult();
        CellState->ResetResult();

        //Hidden state
        BeforeHidden->ResetResult();
        Hidden->ResetResult();

        Tensor<DTYPE> *result = this->GetResult();
        result->Reset();
    }

    int ResetGradient() {
        //time
        m_aTempHidden->ResetGradient();
        m_TempCellState->ResetGradient();

        MatMul_I2G->ResetGradient();
        MatMul_H2G->ResetGradient();
        AddGates->ResetGradient();
        AddBias->ResetGradient();

        //forget Gate
        ForgetGateInput->ResetGradient();
        ForgetGateSigmoid->ResetGradient();

        //Input Gate
        InputGateInput->ResetGradient();
        InputGateSigmoid->ResetGradient();

        //
        CellGateInput->ResetGradient();
        CellGateTanh->ResetGradient();

        //Output Gate
        OutputGateInput->ResetGradient();
        OutputGateSigmoid->ResetGradient();

        //Cell state
        ForgetGateCell->ResetGradient();
        InputGateCell->ResetGradient();
        CellState->ResetGradient();

        //Hidden state
        BeforeHidden->ResetGradient();
        Hidden->ResetGradient();

        Tensor<DTYPE> *grad = this->GetGradient();
        grad->Reset();
    }


};


#endif  // LSTM2_H_
