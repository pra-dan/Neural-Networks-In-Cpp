//Had a lot of trouble with shuffle
//Added linear activation beside tanh
#include <iostream>
#include<vector>
#include <list>
#include <cstdlib>
#include <math.h>

#define PI 3.141592653589793238463
#define N

#define epsilon 0.1
#define epoch 1

using namespace std;
extern "C" FILE *popen(const char *command, const char *mode);

struct HiddenLayer
{
    int nodes;     ///Number of nodes in this HiddenLayer
    float *Wx;
    float *b;
};

struct Synapse
{
    int prev_layer_nodes;     ///Number of Nodes in the left layer
    int next_layer_nodes;     ///Number of Nodes in the right layer
    float** w;                ///Weights of the connections
};
//double sigmoid(double x) { return 1.0f / (1.0f + exp(-x)); }
//double dsigmoid(double x) { return x * (1.0f - x); }
double tanh(double x) { return (exp(x)-exp(-x))/(exp(x)+exp(-x)) ;}
double dtanh(double x) {return 1.0f - x*x ;}

double lin(double x) { return x;}
double dlin(double x) { return 1.0f;}

double init_weight() { return (2*rand()/RAND_MAX -1); }

static const int numInputs = 1;
static int numHiddenLayers;
static int numSynapses;
//static const int numHiddenNodes;
static const int numOutputs;

///CONFIGURE THE NUMBER THE NUMBER OF HIDDEN LAYERS
double configure_NN_HiddenLayers(int n)
{
    numHiddenLayers = n;
    numSynapses = n+1;
    struct HiddenLayer HLayer[n];
}

///CONFIGURE THE NUMBER THE NUMBER OF HIDDEN LAYER NODES
//double configure_NN_HiddenNodes(int n){numHiddenNodes[numHi] = n;}

///CONFIGURE THE NUMBER THE NUMBER OF OUTPUT NODES
void configure_NN_OutputNeurons(int n) {numOutputs = n; float outputLayer[n]; float outputBias[n];}

double MAXX = -9999999999999999; //maximum value of input example
//double init_weight() { return ((double)rand())/((double)RAND_MAX); }

const double lr = 0.05f;
double hiddenLayer[numHiddenNodes];//
double outputLayer[numOutputs];

double hiddenLayerBias[numHiddenNodes];
double outputLayerBias[numOutputs];

//double hiddenWeights[numInputs][numHiddenNodes];
//double outputWeights[numHiddenNodes][numOutputs];

static const int numTrainingSets = 50;
double training_inputs[numTrainingSets][numInputs];
double training_outputs[numTrainingSets][numOutputs];

void shuffle(int *array, size_t n)
{
    if (n > 1) //If no. of training examples > 1
    {
        size_t i;
        for (i = 0; i < n - 1; i++)
        {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

void predict(double test_sample[])
{
    for (int j=0; j<numHiddenNodes; j++)
    {
        double activation=hiddenLayerBias[j];
        for (int k=0; k<numInputs; k++)
        {
            activation+=test_sample[k]*hiddenWeights[k][j];
        }
        hiddenLayer[j] = tanh(activation);
    }

    for (int j=0; j<numOutputs; j++)
    {
        double activation=outputLayerBias[j];
        for (int k=0; k<numHiddenNodes; k++)
        {
            activation+=hiddenLayer[k]*outputWeights[k][j];
        }
        outputLayer[j] = lin(activation);
    }
    //std::cout<<outputLayer[0]<<"\n";
    //return outputLayer[0];
    //std::cout << "Input:" << training_inputs[i][0] << " " << training_inputs[i][1] << "    Output:" << outputLayer[0] << "    Expected Output: " << training_outputs[i][0] << "\n";
}

int main(int argc, const char * argv[])
{
    /********************************************************************/
    configure_NN_HiddenLayers(1);
    configure_NN_HiddenNodes(5);
    configure_NN_OutputNeurons(1);

    ///Initialize all the synapse connections between input layer and 1st hidden layer
    struct Synapse syn[numSynapses]; //synapse between input-hidden and hidden-output
    syn[0].prev_layer_nodes = 1;
    syn[0].next_layer_nodes = 5;
    syn[0].w = (float **)malloc(syn[0].prev_layer_nodes * sizeof(float *));
    for(int i=0; i<syn[0].prev_layer_nodes; i++)
     {
         //Allocating memory
         syn[0].w[i] = (float *)malloc(syn[0].next_layer_nodes * sizeof(float));
         //Initializing weights
         syn[0].w[i] = init_weight();
     }

    ///Initialize all the nodes and biases in the 1st hidden layer
    HLayer[0].nodes = syn[0].next_layer_nodes;
    //Allocating memory
    HLayer[0].Wx = (float *)malloc(HLayer[0].nodes * sizeof(float));
    HLayer[0].b = (float *)malloc(HLayer[0].nodes * sizeof(float));
    //Initializing weights
    for(int i=0; i<HLayer[0].nodes; ++i)
    {
      HLayer[0].Wx[i] =  init_weight();
      HLayer[0].b[i] =  init_weight();
    }

    ///Initialize all the synapse connections between hidden layer and output layer
    syn[1].prev_layer_nodes = 5;
    syn[1].next_layer_nodes = 1;
    syn[1].w = (float **)malloc(syn[1].prev_layer_nodes * sizeof(float *));
    for(int i=0; i<syn[1].prev_layer_nodes; i++)
     {
         //Allocating memory
         syn[1].w[i] = (float *)malloc(syn[1].next_layer_nodes * sizeof(float));
         //Initializing weights
         syn[1].w[i] = init_weight();
     }
    /***************************************************************/
    ///Initialize the output neurons
    for(int i=0; i<numOutputs; ++i)
    {
        OutputWeight[i] = init_weight();
        OutputBias[i] = init_weight();
    }

    ///TRAINING DATA GENERATION
    for (int i = 0; i < numTrainingSets; i++)
    {
		double p = (2*PI*(double)i/numTrainingSets);
		training_inputs[i][0] = (p);
		training_outputs[i][0] = sin(p);

		/***************************Try Avoiding Edits In This part*******************************/
        ///FINDING NORMALIZING FACTOR
        for(int m=0; m<numInputs; ++m)
            if(MAXX < training_inputs[i][m])
                MAXX = training_inputs[i][m];
        for(int m=0; m<numOutputs; ++m)
            if(MAXX < training_outputs[i][m])
                MAXX = training_outputs[i][m];
	}

	///NORMALIZING
	for (int i = 0; i < numTrainingSets; i++)
	{
        for(int m=0; m<numInputs; ++m)
            training_inputs[i][m] /= 1.0f*MAXX;

        for(int m=0; m<numOutputs; ++m)
            training_outputs[i][m] /= 1.0f*MAXX;

        cout<<"In: "<<training_inputs[i][0]<<"  out: "<<training_outputs[i][0]<<endl;
	}
    ///WEIGHT & BIAS INITIALIZATION
    /*
    struct Synapse hiddenWeights[numHiddenLayers];
    //Synapse 0
    hiddenWeights[0].prev = numInputs;
    hiddenWeights[0].next = HLayer[0];
    hiddenWeights[0].node = (float*)malloc(sizeof(float) * hiddenWeights[0].prev * hiddenWeights[0].next);
    */

    /*
    for (int i=0; i<numInputs; i++) {
        for (int j=0; j<numHiddenNodes; j++) {
            hiddenWeights[i][j] = init_weight();
        }
    }
    for (int i=0; i<numHiddenNodes; i++) {
        hiddenLayerBias[i] = init_weight();
        for (int j=0; j<numOutputs; j++) {
            outputWeights[i][j] = init_weight();
        }
    }
    for (int i=0; i<numOutputs; i++) {
        //outputLayerBias[i] = init_weight();
        outputLayerBias[i] = 0;
    }
    */
    ///FOR INDEX SHUFFLING
    int trainingSetOrder[numTrainingSets];
    for(int j=0; j<numInputs; ++j)
        trainingSetOrder[j] = j;


    ///TRAINING
    //std::cout<<"start train\n";
    vector<double> performance, epo; ///STORE MSE, EPOCH
    for (int n=0; n < epoch; n++)
    {
        double MSE = 0;
        shuffle(trainingSetOrder,numTrainingSets);
        std::cout<<"epoch :"<<n<<"\n";
        for (int i=0; i<numTrainingSets; i++)
        {
            //int i = trainingSetOrder[x];
            int x=i;
            //std::cout<<"Training Set :"<<x<<"\n";
            /// Forward pass
            /*
            for (int j=0; j<numHiddenNodes; j++)
            {
                double activation=hiddenLayerBias[j];
                //std::cout<<"Training Set :"<<x<<"\n";
                 for (int k=0; k<numInputs; k++) {
                    activation+=training_inputs[x][k]*hiddenWeights[k][j];
                }
                hiddenLayer[j] = tanh(activation);
            }

            for (int j=0; j<numOutputs; j++) {
                double activation=outputLayerBias[j];
                for (int k=0; k<numHiddenNodes; k++)
                {
                    activation+=hiddenLayer[k]*outputWeights[k][j];
                }
                outputLayer[j] = lin(activation);
            }

            */
            ///CONSIDER THE SYNAPSE-LAYER PAIR... WE'LL BE PROCESSING THEM AS A BATCH
            ///...EXCEPT FOR THE LAST_SYNAPSE-OUTPUT_LAYER PAIR
            int i = 0;
            {
               for(int j=0; j<syn[i].next_layer_nodes; ++j)
               {
                   double activation = HLayer[i].b[j];
                   for(int k=0; k<syn[i].prev_layer_nodes; ++k)
                   {
                       activation += training_inputs[x][k]*syn[i].w[k][j];
                   }
                   HLayer[i].Wx[j] = tanh(activation);
               }
            }
            ///PHASE 2: CONSIDER THE REMAINING SYNAPSE-HIDDEN LAYER PAIRS
            bool phase2_entry =false;
            for(; i<numHiddenLayers; ++i)
            {
                for(int j=0; j<syn[i].next_layer_nodes; ++j)
                {
                   double activation = HLayer[i].b[j];
                   for(int k=0; k<syn[i].prev_layer_nodes; ++k)
                   {
                       activation += HLayer[i-1].Wx[k] * syn[i].w[k][j];
                   }
                   HLayer[i].Wx[j] = tanh(activation);
                }
                phase2_entry = true;
            }
            ///PHASE 3: CONSIDER THE LAST_SYNAPSE-OUTPUT_LAYER PAIR
            if(phase2_entry == true)
                i--; //Using the previous synapse

            {
                i++;  //Consider the last synapse
                for(int j=0; j<numOutputs; ++j)
                {
                   double activation = outputBias[j];
                   for(int k=0; k<syn[i].prev_layer_nodes; ++k)     //The k iterates over num of nodes in last hidden layer
                   {
                       activation += HLayer[i-1].Wx[k] * syn[i].w[k][j];
                   }
                   outputLayer[j] = lin(activation);
                }
            }
            //std::cout << "Input:" << training_inputs[x][0] << " " << "    Output:" << outputLayer[0] << "    Expected Output: " << training_outputs[x][0] << "\n";
            for(int k=0; k<numOutputs; ++k)
                MSE += (1.0f/numOutputs)*pow( training_outputs[x][k] - outputLayer[k], 2);

           /// Backprop
           ///   For V
            double deltaOutput[numOutputs];
            for (int j=0; j<numOutputs; j++)
            {
                double errorOutput = (training_outputs[i][j]-outputLayer[j]);
                deltaOutput[j] = errorOutput*dlin(outputLayer[j]);
            }
            /**Consider (nth) Layer-(n-1 th) Synapse pairs
            *For W(n), W(n-1), W(n-2)...
            **PHASE A
            ***Only for Output Layer-Last Synapse pair*/
            int i=numSynapses-1; //Beginning with the last synapse
            {
                for(int j=0; j<HLayer[i].nodes; j++)
                {
                    float errorHidden = 0.0f;
                    for(int k=0; k<numOutputs; k++)
                    {
                        errorHidden += deltaOutput[k] * syn[i].w[j][k];
                    }
                }
            }
            for(; i>0; --i) //Iterating in reverse fashion


            ///Updation
            ///   For V and b
            for (int j=0; j<numOutputs; j++) {
                //b
                outputLayerBias[j] += deltaOutput[j]*lr;
                for (int k=0; k<numHiddenNodes; k++)
                {
                    outputWeights[k][j]+= hiddenLayer[k]*deltaOutput[j]*lr;
                }
            }

            ///   For W and c
            for (int j=0; j<numHiddenNodes; j++) {
                //c
                hiddenLayerBias[j] += deltaHidden[j]*lr;
                //W
                for(int k=0; k<numInputs; k++) {
                  hiddenWeights[k][j]+=training_inputs[i][k]*deltaHidden[j]*lr;
                }
            }
        }
        //Averaging the MSE
        MSE /= 1.0f*numTrainingSets;
        //cout<< "  MSE: "<< MSE<<endl;
        ///Steps to PLOT PERFORMANCE PER EPOCH
        performance.push_back(MSE*100);
        epo.push_back(n);
    }

    // Print weights
    std::cout << "Final Hidden Weights\n[ ";
    for (int j=0; j<numHiddenNodes; j++) {
        std::cout << "[ ";
        for(int k=0; k<numInputs; k++) {
            std::cout << hiddenWeights[k][j] << " ";
        }
        std::cout << "] ";
    }
    std::cout << "]\n";

    std::cout << "Final Hidden Biases\n[ ";
    for (int j=0; j<numHiddenNodes; j++) {
        std::cout << hiddenLayerBias[j] << " ";

    }
    std::cout << "]\n";
    std::cout << "Final Output Weights";
    for (int j=0; j<numOutputs; j++) {
        std::cout << "[ ";
        for (int k=0; k<numHiddenNodes; k++) {
            std::cout << outputWeights[k][j] << " ";
        }
        std::cout << "]\n";
    }
    std::cout << "Final Output Biases\n[ ";
    for (int j=0; j<numOutputs; j++) {
        std::cout << outputLayerBias[j] << " ";
    }
    std::cout << "]\n";

    //Plot the results
	vector<float> x;
	vector<float> y1, y2;
    //double test_input[1000][numInputs];
    int numTestSets = numTrainingSets;
	for (float i = 0; i < numTestSets; i=i+0.25)
	{
        double p = (2*PI*(double)i/numTestSets);
		x.push_back(p);
		y1.push_back(sin(p));
		double test_input[1];
		test_input[0] = p/MAXX;
        predict(test_input);
		y2.push_back(outputLayer[0]*MAXX);
	}
    /*
	FILE * gp = popen("gnuplot", "w");
	fprintf(gp, "set terminal wxt size 600,400 \n");
	fprintf(gp, "set grid \n");
	fprintf(gp, "set title '%s' \n", "f(x) = x sin (x)");
	fprintf(gp, "set style line 1 lt 3 pt 7 ps 0.1 lc rgb 'green' lw 1 \n");
	fprintf(gp, "set style line 2 lt 3 pt 7 ps 0.1 lc rgb 'red' lw 1 \n");
	fprintf(gp, "plot '-' w p ls 1, '-' w p ls 2 \n");

	///Exact f(x) = sin(x) -> Green Graph
	for (int k = 0; k < x.size(); k++) {
		fprintf(gp, "%f %f \n", x[k], y1[k]);
	}
	fprintf(gp, "e\n");

	///Neural Network Approximate f(x) = xsin(x) -> Red Graph
	for (int k = 0; k < x.size(); k++) {
		fprintf(gp, "%f %f \n", x[k], y2[k]);
	}
	fprintf(gp, "e\n");

	fflush(gp);
    ///FILE POINTER FOR SECOND PLOT (PERFORMANCE GRAPH)
    FILE * gp1 = popen("gnuplot", "w");
	fprintf(gp1, "set terminal wxt size 600,400 \n");
	fprintf(gp1, "set grid \n");
	fprintf(gp1, "set title '%s' \n", "Performance");
	fprintf(gp1, "set style line 1 lt 3 pt 7 ps 0.1 lc rgb 'green' lw 1 \n");
	fprintf(gp1, "set style line 2 lt 3 pt 7 ps 0.1 lc rgb 'red' lw 1 \n");
	fprintf(gp1, "plot '-' w p ls 1 \n");

    for (int k = 0; k < epo.size(); k++) {
		fprintf(gp1, "%f %f \n", epo[k], performance[k]);
	}
	fprintf(gp1, "e\n");

	fflush(gp1);

	system("pause");
	//_pclose(gp);
    */
    return 0;
}
