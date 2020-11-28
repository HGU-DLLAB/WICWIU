#ifndef __EMBEDDING_LAYER__
#define __EMBEDDING_LAYER__    value

#include "../Module.hpp"


template<typename DTYPE> class EmbeddingLayer : public Module<DTYPE>{
private:
public:

    EmbeddingLayer(Operator<DTYPE> *pInput, int vocabsize, int embeddingDim, std::string pName = "No Name") : Module<DTYPE>(pName) {
        Alloc(pInput, vocabsize, embeddingDim, pName);
    }

    virtual ~EmbeddingLayer() {}


    int Alloc(Operator<DTYPE> *pInput, int vocabsize, int embeddingDim, std::string pName) {
        this->SetInput(pInput);

        Operator<DTYPE> *out = pInput;

        Tensorholder<DTYPE> *pWeight_embedding = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, vocabsize, embeddingDim, 0.0, 0.01), "EmbedLayer_pWeight" + pName);


        out = new Embedding<DTYPE>(pWeight_embedding, out, "Embedding_Operator");

        this->AnalyzeGraph(out);

        return TRUE;
    }

};


#endif  // __EMBEDDING_LAYER__
