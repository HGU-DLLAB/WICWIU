#ifndef LSTM_H_
#define LSTM_H_    value

#include "../Operator.hpp"
#include "MatMul.hpp"
#include "Add.hpp"
#include "Tanh.hpp"
#include "Tensorholder.hpp"

#define TIME     0
#define BATCH    1

template<typename DTYPE> class LSTM : public Operator<DTYPE>{
private:

    //forget Gate
    Operator<DTYPE> *m_aInput2ForgetGates;
    Operator<DTYPE> *m_aHIdden2ForgetGates;
    Operator<DTYPE> *BeforeForgetGates_a;
    Operator<DTYPE> *ForgetGates_a;
    Operator<DTYPE> *ForgetGates_b;

    //Input Gate
    Operator<DTYPE> *m_aInput2InputGates;
    Operator<DTYPE> *m_aHidden2InputGates;
    Operator<DTYPE> *BeforeInputGates_a;
    Operator<DTYPE> *InputGates_a;
    Operator<DTYPE> *InputGates_b;

    //
    Operator<DTYPE> *m_aInput2Cell;
    Operator<DTYPE> *m_aHidden2Cell;
    Operator<DTYPE> *BeforeCell_a;
    Operator<DTYPE> *Cell_a;
    Operator<DTYPE> *g;

    Operator<DTYPE> *InputGateAndG;

    //Output Gate
    Operator<DTYPE> *m_aInput2OutputGates;
    Operator<DTYPE> *m_aHidden2OutputGates;
    Operator<DTYPE> *BeforeOutputGates_a;
    Operator<DTYPE> *OutputGates_a;
    Operator<DTYPE> *OutputGates_b;

    //Cell state
    Operator<DTYPE> *ForgetCell;
    Operator<DTYPE> *CellState;

    //Hidden state
    Operator<DTYPE> *BeforeHidden;
    Operator<DTYPE> *Hidden;

    //time 처리
    Operator<DTYPE> *m_aTempHidden;
    Operator<DTYPE> *m_TempCellState;


public:
  LSTM(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightIF, Operator<DTYPE> *pWeightHF, Operator<DTYPE> *pWeightII, Operator<DTYPE> *pWeightHI, Operator<DTYPE> *pWeightIC, Operator<DTYPE> *pWeightHC, Operator<DTYPE> *pWeightIO, Operator<DTYPE> *pWeightHO,
       Operator<DTYPE> *fBias, Operator<DTYPE> *iBias, Operator<DTYPE> *cBias, Operator<DTYPE> *oBias)
       : Operator<DTYPE>(13, pInput, pWeightIF, pWeightHF, pWeightII, pWeightHI, pWeightIC, pWeightHC, pWeightIO, pWeightHO, fBias, iBias, cBias, oBias) {
      #if __DEBUG__
      std::cout << "LSTM::LSTM(Operator<DTYPE> *)" << '\n';
      #endif  // __DEBUG__
      this->Alloc(pInput, pWeightIF, pWeightHF, pWeightII, pWeightHI, pWeightIC, pWeightHC, pWeightIO, pWeightHO,
           fBias, iBias, cBias, oBias);
  }

    LSTM(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightIF, Operator<DTYPE> *pWeightHF, Operator<DTYPE> *pWeightII, Operator<DTYPE> *pWeightHI, Operator<DTYPE> *pWeightIC, Operator<DTYPE> *pWeightHC, Operator<DTYPE> *pWeightIO, Operator<DTYPE> *pWeightHO,
         Operator<DTYPE> *fBias, Operator<DTYPE> *iBias, Operator<DTYPE> *cBias, Operator<DTYPE> *oBias, std::string pName)
         : Operator<DTYPE>(13, pInput, pWeightIF, pWeightHF, pWeightII, pWeightHI, pWeightIC, pWeightHC, pWeightIO, pWeightHO, fBias, iBias, cBias, oBias) {
        #if __DEBUG__
        std::cout << "LSTM::LSTM(Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, pWeightIF, pWeightHF, pWeightII, pWeightHI, pWeightIC, pWeightHC, pWeightIO, pWeightHO,
             fBias, iBias, cBias, oBias);
    }

    ~LSTM() {
        #if __DEBUG__
        std::cout << "LSTM::~LSTM()" << '\n';
        #endif  // __DEBUG__

        Delete();
    }

    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightIF, Operator<DTYPE> *pWeightHF, Operator<DTYPE> *pWeightII, Operator<DTYPE> *pWeightHI, Operator<DTYPE> *pWeightIC, Operator<DTYPE> *pWeightHC, Operator<DTYPE> *pWeightIO, Operator<DTYPE> *pWeightHO,
              Operator<DTYPE> *fBias, Operator<DTYPE> *iBias, Operator<DTYPE> *cBias, Operator<DTYPE> *oBias) {

        Shape *InputShape    = pInput->GetResult()->GetShape();
        Shape *WeightIHShape = pWeightIF->GetResult()->GetShape();

        int hidTimeSize  = (*InputShape)[TIME];
        int hidBatchSize = (*InputShape)[BATCH];
        int hidColSize   = (*WeightIHShape)[3];


        //time
        m_aTempHidden         = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(hidTimeSize, hidBatchSize, 1, 1, hidColSize), "tempHidden");
        m_TempCellState       = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(hidTimeSize, hidBatchSize, 1, 1, hidColSize), "tempCell");


        //forget Gate
        m_aInput2ForgetGates  = new MatMul<DTYPE>(pWeightIF, pInput, "lstm_matmul_IF");
        m_aHIdden2ForgetGates = new MatMul<DTYPE>(pWeightHF, m_aTempHidden, "lstm_matmul_HF");
        BeforeForgetGates_a   = new Addall<DTYPE>(m_aInput2ForgetGates, m_aHIdden2ForgetGates, "lstm_F_addall");
        ForgetGates_a         = new AddColWise<DTYPE>(BeforeForgetGates_a, fBias, "lstm_F_addall");
        ForgetGates_b         = new Sigmoid<DTYPE>(ForgetGates_a, "lstm_f_sigmoid");

        //Input Gate
        m_aInput2InputGates   = new MatMul<DTYPE>(pWeightII, pInput, "lstm_matmul_II");
        m_aHidden2InputGates  = new MatMul<DTYPE>(pWeightHI, m_aTempHidden, "lstm_matmul_HI");
        BeforeInputGates_a    = new Addall<DTYPE>(m_aInput2InputGates, m_aHidden2InputGates, "lstm_I_addall");
        InputGates_a          = new AddColWise<DTYPE>(BeforeInputGates_a, iBias, "lstm_I_addall");
        InputGates_b          = new Sigmoid<DTYPE>(InputGates_a, "lstm_I_sigmoid");

        //
        m_aInput2Cell         = new MatMul<DTYPE>(pWeightIC, pInput, "lstm_matmul_IC");
        m_aHidden2Cell        = new MatMul<DTYPE>(pWeightHC, m_aTempHidden, "lstm_matmul_HC");
        BeforeCell_a          = new Addall<DTYPE>(m_aInput2Cell, m_aHidden2Cell, "lstm_C_addall");
        Cell_a                = new AddColWise<DTYPE>(BeforeCell_a, cBias, "lstm_c_addall");
        g                     = new Tanh<DTYPE>(Cell_a, "lstm_c_tanh");


        //Output Gate
        m_aInput2OutputGates     = new MatMul<DTYPE>(pWeightIO, pInput, "lstm_matmul_IO");
        m_aHidden2OutputGates    = new MatMul<DTYPE>(pWeightHO, m_aTempHidden, "lstm_matmul_IO");
        BeforeOutputGates_a      = new Addall<DTYPE>(m_aInput2OutputGates, m_aHidden2OutputGates, "lstm_O_addall");
        OutputGates_a            = new AddColWise<DTYPE>(BeforeOutputGates_a, oBias, "lstm_o_addall");
        OutputGates_b            = new Sigmoid<DTYPE>(OutputGates_a, "lstm_o_sigmoid");

        //Cell state
        ForgetCell            = new Hadamard<DTYPE>(m_TempCellState, ForgetGates_b, "forgetcell");
        InputGateAndG         = new Hadamard<DTYPE>(g, InputGates_b, "beforecellstate");
        CellState             = new Addall<DTYPE>(ForgetCell, InputGateAndG, "cellState");

        //Hidden state
        BeforeHidden          = new Tanh<DTYPE>(CellState, "beforehidden");
        Hidden                = new Hadamard<DTYPE>(BeforeHidden, OutputGates_b, "cellstate");


        //For AnalyzeGraph
        //input
        pInput->GetOutputContainer()->Pop(m_aInput2ForgetGates);
        pInput->GetOutputContainer()->Pop(m_aInput2InputGates);
        pInput->GetOutputContainer()->Pop(m_aInput2Cell);
        pInput->GetOutputContainer()->Pop(m_aInput2OutputGates);

        //weight
        pWeightIF->GetOutputContainer()->Pop(m_aInput2ForgetGates);
        pWeightHF->GetOutputContainer()->Pop(m_aHIdden2ForgetGates);

        pWeightII->GetOutputContainer()->Pop(m_aInput2InputGates);
        pWeightHI->GetOutputContainer()->Pop(m_aHidden2InputGates);

        pWeightIC->GetOutputContainer()->Pop(m_aInput2Cell);
        pWeightHC->GetOutputContainer()->Pop(m_aHidden2Cell);

        pWeightIO->GetOutputContainer()->Pop(m_aInput2OutputGates);
        pWeightHO->GetOutputContainer()->Pop(m_aHidden2OutputGates);

        //biad
        fBias->GetOutputContainer()->Pop(ForgetGates_a);
        iBias->GetOutputContainer()->Pop(InputGates_a);
        cBias->GetOutputContainer()->Pop(Cell_a);
        oBias->GetOutputContainer()->Pop(OutputGates_a);



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

            int colSize        = prevHidden->GetColSize();
            Shape *HiddenShape = prevHidden->GetShape();

            for (int i = 0; i < colSize; i++) {
                (*tempHidden)[Index5D(HiddenShape, pTime, 0, 0, 0, i)] = (*prevHidden)[Index5D(HiddenShape, pTime - 1, 0, 0, 0, i)];
            }

            //cell state
            Tensor<DTYPE> *prevCellState = CellState->GetResult();
            Tensor<DTYPE> *tempCellState = m_TempCellState->GetResult();

            colSize        = prevCellState->GetColSize();
            Shape *CellShape = prevCellState->GetShape();

            for (int i = 0; i < colSize; i++) {
                (*tempCellState)[Index5D(CellShape, pTime, 0, 0, 0, i)] = (*prevCellState)[Index5D(CellShape, pTime - 1, 0, 0, 0, i)];
            }

        }


        //Forget Gates
        m_aInput2ForgetGates->ForwardPropagate(pTime);
        m_aHIdden2ForgetGates->ForwardPropagate(pTime);
        BeforeForgetGates_a->ForwardPropagate(pTime);
        ForgetGates_a->ForwardPropagate(pTime);
        ForgetGates_b->ForwardPropagate(pTime);

        //Input Gates
        m_aInput2InputGates->ForwardPropagate(pTime);
        m_aHidden2InputGates->ForwardPropagate(pTime);
        BeforeInputGates_a->ForwardPropagate(pTime);
        InputGates_a->ForwardPropagate(pTime);
        InputGates_b->ForwardPropagate(pTime);


        //
        m_aInput2Cell->ForwardPropagate(pTime);
        m_aHidden2Cell->ForwardPropagate(pTime);
        BeforeCell_a->ForwardPropagate(pTime);
        Cell_a->ForwardPropagate(pTime);
        g->ForwardPropagate(pTime);

        //Output Gate
        m_aInput2OutputGates->ForwardPropagate(pTime);
        m_aHidden2OutputGates->ForwardPropagate(pTime);
        BeforeOutputGates_a->ForwardPropagate(pTime);
        OutputGates_a->ForwardPropagate(pTime);
        OutputGates_b->ForwardPropagate(pTime);

        //Cell state
        ForgetCell->ForwardPropagate(pTime);
        InputGateAndG->ForwardPropagate(pTime);
        CellState->ForwardPropagate(pTime);

        //Hidden state
        BeforeHidden->ForwardPropagate(pTime);
        Hidden->ForwardPropagate(pTime);


        Tensor<DTYPE> *_result = Hidden->GetResult();
        Tensor<DTYPE> *result  = this->GetResult();

        int colSize        = result->GetColSize();
        Shape *ResultShape = result->GetShape();

        for (int i = 0; i < colSize; i++) {
            (*result)[Index5D(ResultShape, pTime, 0, 0, 0, i)] = (*_result)[Index5D(ResultShape, pTime, 0, 0, 0, i)];
        }

        return TRUE;
    }


    int BackPropagate(int pTime = 0) {

        Tensor<DTYPE> *_grad = Hidden->GetGradient();
        Tensor<DTYPE> *grad  = this->GetGradient();

        int colSize        = grad->GetColSize();
        int timeSize       = grad->GetTimeSize();
        Shape *ResultShape = grad->GetShape();

        for (int i = 0; i < colSize; i++) {
            (*_grad)[Index5D(ResultShape, pTime, 0, 0, 0, i)] = (*grad)[Index5D(ResultShape, pTime, 0, 0, 0, i)];
        }


        if (pTime != timeSize-1) {

            Tensor<DTYPE> *tempHiddenGrad = m_aTempHidden->GetGradient();
            Tensor<DTYPE> *prevHiddenGrad = Hidden->GetGradient();

            int colSize        = tempHiddenGrad->GetColSize();
            Shape *HiddenShape = tempHiddenGrad->GetShape();

            for (int i = 0; i < colSize; i++) {
                (*prevHiddenGrad)[Index5D(HiddenShape, pTime, 0, 0, 0, i)] += (*tempHiddenGrad)[Index5D(HiddenShape, pTime+1, 0, 0, 0, i)];
            }
        }

        //Hidden state
        Hidden->BackPropagate(pTime);
        BeforeHidden->BackPropagate(pTime);

        //Cell state
          if (pTime != timeSize-1) {

              Tensor<DTYPE> *tempCellGrad = m_TempCellState->GetGradient();
              Tensor<DTYPE> *prevCellGrad = CellState->GetGradient();

              int colSize        = tempCellGrad->GetColSize();
              Shape *CellShape = tempCellGrad->GetShape();

              for (int i = 0; i < colSize; i++) {
                  (*prevCellGrad)[Index5D(CellShape, pTime, 0, 0, 0, i)] += (*tempCellGrad)[Index5D(CellShape, pTime+1, 0, 0, 0, i)];
              }
          }

        CellState->BackPropagate(pTime);
        InputGateAndG->BackPropagate(pTime);
        ForgetCell->BackPropagate(pTime);

        //Output Gate
        OutputGates_b->BackPropagate(pTime);
        OutputGates_a->BackPropagate(pTime);
        BeforeOutputGates_a->BackPropagate(pTime);
        m_aHidden2OutputGates->BackPropagate(pTime);
        m_aInput2OutputGates->BackPropagate(pTime);

        //
        g->BackPropagate(pTime);
        Cell_a->BackPropagate(pTime);
        BeforeCell_a->BackPropagate(pTime);
        m_aHidden2Cell->BackPropagate(pTime);
        m_aInput2Cell->BackPropagate(pTime);

        //Input Gates
        InputGates_b->BackPropagate(pTime);
        InputGates_a->BackPropagate(pTime);
        BeforeInputGates_a->BackPropagate(pTime);
        m_aHidden2InputGates->BackPropagate(pTime);
        m_aInput2InputGates->BackPropagate(pTime);

        //Forget Gates
        ForgetGates_b->BackPropagate(pTime);
        ForgetGates_a->BackPropagate(pTime);
        BeforeForgetGates_a->BackPropagate(pTime);
        m_aHIdden2ForgetGates->BackPropagate(pTime);
        m_aInput2ForgetGates->BackPropagate(pTime);



        return TRUE;
    }


    int ResetResult() {
        //time
        m_aTempHidden->ResetResult();
        m_TempCellState->ResetResult();

        //forget Gate
        m_aInput2ForgetGates->ResetResult();
        m_aHIdden2ForgetGates->ResetResult();
        BeforeForgetGates_a->ResetResult();
        ForgetGates_a->ResetResult();
        ForgetGates_b->ResetResult();

        //Input Gate
        m_aInput2InputGates->ResetResult();
        m_aHidden2InputGates->ResetResult();
        BeforeInputGates_a->ResetResult();
        InputGates_a->ResetResult();
        InputGates_b->ResetResult();

        m_aInput2Cell->ResetResult();
        m_aHidden2Cell->ResetResult();
        BeforeCell_a->ResetResult();
        Cell_a->ResetResult();
        g->ResetResult();

        //Output Gate
        m_aInput2OutputGates->ResetResult();
        m_aHidden2OutputGates->ResetResult();
        BeforeOutputGates_a->ResetResult();
        OutputGates_a->ResetResult();
        OutputGates_b->ResetResult();

        //Cell state
        ForgetCell->ResetResult();
        InputGateAndG->ResetResult();
        CellState->ResetResult();

        //Hidden state
        BeforeHidden->ResetResult();
        Hidden->ResetResult();
    }

    int ResetGradient() {
        //time
        m_aTempHidden->ResetGradient();
        m_TempCellState->ResetGradient();

        //forget Gate
        m_aInput2ForgetGates->ResetGradient();
        m_aHIdden2ForgetGates->ResetGradient();
        BeforeForgetGates_a->ResetGradient();
        ForgetGates_a->ResetGradient();
        ForgetGates_b->ResetGradient();

        //Input Gate
        m_aInput2InputGates->ResetGradient();
        m_aHidden2InputGates->ResetGradient();
        BeforeInputGates_a->ResetGradient();
        InputGates_a->ResetGradient();
        InputGates_b->ResetGradient();

        m_aInput2Cell->ResetGradient();
        m_aHidden2Cell->ResetGradient();
        BeforeCell_a->ResetGradient();
        Cell_a->ResetGradient();
        g->ResetGradient();

        //Output Gate
        m_aInput2OutputGates->ResetGradient();
        m_aHidden2OutputGates->ResetGradient();
        BeforeOutputGates_a->ResetGradient();
        OutputGates_a->ResetGradient();
        OutputGates_b->ResetGradient();

        //Cell state
        ForgetCell->ResetGradient();
        InputGateAndG->ResetGradient();
        CellState->ResetGradient();

        //Hidden state
        BeforeHidden->ResetGradient();
        Hidden->ResetGradient();
    }


};


#endif  // LSTM_H_
