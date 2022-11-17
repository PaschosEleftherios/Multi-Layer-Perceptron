#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#define trainValues 4000
#define testValues 4000
#define dimension 2
#define d 2 //# of inputs
#define K 4 //# of categories
#define H1 15 //#of neurons first level
#define H2 15 //#of neurons second level
#define H3 15 //#of neurons third level
#define h 0.001 //learning rate
#define whatFunc 1 //What function will be used in the hidden levels 0 for relu 1 for tanh
#define errorDiff 10 //threshold for ending training
#define B 40

//Ylopoioun to Dataset S1.
void createDataset();
void printDataset();
void createArrays();
void setNoise();

void setWeights();//give random values on neuron weights
void forward_pass(float *x, int D, float *y, int k);
void backprop(float *x, int D, float *t, int k);
float computeInerProduct(float *x, float *y, int dmnsion);


void gradient_descent();
void setNewWeights();
void setZeros();
void computeGeneralization();
int squareError();



float relu(float u);
float d_relu(float u);
float tanhFunc(float u);
float d_tanh(float u);
float sigmoid(float u);
float d_sigmoid(float u);


//weights and biases
float firstLevelWeights[H1][d];
float secondLevelWeights[H2][H1];
float thirdLevelWeights[K][H3];
float firstLevelPolwsh[H1];
float secondLevelPolwsh[H2];
float thirdLevelPolwsh[K];
float thirdLevelWeightsNew[H3][H2];
float thirdLevelPolwshNew[H3];

//Arrays for the training and testing of the network
float datasetArray[trainValues][d];
float T[trainValues][K];
float Ycomplete[trainValues][K];//for the squareError
float datasetArrayTest[testValues][d];//for checking 
float Ttest[testValues][K];//For testing the network will be used in Generalization
float Ytest[testValues][K];//For testing the network will be used in Generalization

//Arrays for reflecting the Neurons of the network 
//and the neurons after their activation fuction(...*Complete).
float firstLevelNeurons[H1];
float firstLevelNeuronsComplete[H1];
float secondLevelNeurons[H2];
float secondLevelNeuronsComplete[H2];
float thirdLevelNeurons[K];
float thirdLevelNeuronsComplete[K];
float thirdLevelNeuronsNew[H3];
float thirdLevelNeuronsCompleteNew[H3];

//delta of every level
float deltaOut[K];
float deltaSecondLevel[H2];
float deltaFirstLevel[H1];
float deltaThirdLevel[H3];

//partial change of every level
float deh1[H1][d];
float deh1polwsh[H1];
float deh2[H2][H1];
float deh2polwsh[H2];
float deOut[K][H3];
float deOutPolwsh[K];
float deh3[H3][H2];
float deh3polwsh[H3];

//complete change
float deh1Complete[H1][d];
float deh1polwshComplete[H1];
float deh2Complete[H2][H1];
float deh2polwshComplete[H2];
float deOutComplete[K][H3];
float deOutPolwshComplete[K];
float deh3Complete[H3][H2];
float deh3polwshComplete[H3];

int main(){
    int epoxes=0;
    float sfalma=0.0,sfalmaBefore=0.0;
    srand(time(NULL));
    //createDataset();//Makes A new dataset (a new Train file and a new Test File)
    createArrays();
    setNoise();
    setWeights();
    do{
        gradient_descent();
        sfalmaBefore=sfalma;
        sfalma=squareError();
        printf("Epoxh : %d \t Sfalma: %f\n",epoxes,sfalma);
        epoxes++;
        if (epoxes>700){
            if (abs(sfalma-sfalmaBefore)<=errorDiff && sfalma<=240){
                break;
            }
        }
    }while(1);
    computeGeneralization();
    return 0;
}

void createDataset(){
//Create the Dataset 2.
//First Category
    int i=0;
    FILE *fp1,*fp2;
    fp1 = fopen("DatasetTrain.txt","w");//Make the file of the Dataset
    fp2 = fopen("DatasetTest.txt","w");//Make the file of the Dataset
    float con1,con2,con3,con4 = 0;
    srand(time(NULL));
    for (i=0;i<4000;i++){
        float x1 = (float)(rand() % (100 + 1 - (-100)) + (-100))/100;
        float x2 = (float)(rand() % (100 + 1 - (-100)) + (-100))/100;
        con1 = pow(x1-0.5,2) + pow(x2-0.5,2); //check if it's in category 1
        if(con1<0.16){
            fprintf(fp1,"%1.3f %1.3f 1 0 0 0\n", x1, x2);
            continue;
        }
        con2 = pow(x1+0.5,2) + pow(x2+0.5,2); //check if it's in category 1
        if(con2<0.16){
            fprintf(fp1,"%1.3f %1.3f 1 0 0 0\n", x1, x2);
            continue;
        }
        con3 = pow(x1-0.5,2) + pow(x2+0.5,2); //check if it's in category 2
        if(con3<0.16){
            fprintf(fp1,"%1.3f %1.3f 0 1 0 0\n", x1, x2);
            continue;
        }
        con4 = pow(x1+0.5,2) + pow(x2-0.5,2); //check if it's in category 2
        if(con4<0.16){
            fprintf(fp1,"%1.3f %1.3f 0 1 0 0\n", x1, x2);
            continue;
        }
        if((x1>=0 && x2>=0) || (x1<=0 && x2<=0)){
            fprintf(fp1,"%1.3f %1.3f 0 0 1 0\n", x1, x2); //check if it's in category 3
            continue;
        }else{
            fprintf(fp1,"%1.3f %1.3f 0 0 0 1\n", x1, x2); //check if it's in category 4
        }
    }
    //Make the Test files.
    for (i=0;i<4000;i++){
        float x1 = (float)(rand() % (100 + 1 - (-100)) + (-100))/100;
        float x2 = (float)(rand() % (100 + 1 - (-100)) + (-100))/100;
        con1 = pow(x1-0.5,2) + pow(x2-0.5,2); //check if it's in category 1
        if(con1<0.16){
            fprintf(fp2,"%1.3f %1.3f 1 0 0 0\n", x1, x2);
            continue;
        }
        con2 = pow(x1+0.5,2) + pow(x2+0.5,2); //check if it's in category 1
        if(con2<0.16){
            fprintf(fp2,"%1.3f %1.3f 1 0 0 0\n", x1, x2);
            continue;
        }
        con3 = pow(x1-0.5,2) + pow(x2+0.5,2); //check if it's in category 2
        if(con3<0.16){
            fprintf(fp2,"%1.3f %1.3f 0 1 0 0\n", x1, x2);
            continue;
        }
        con4 = pow(x1+0.5,2) + pow(x2-0.5,2); //check if it's in category 2
        if(con4<0.16){
            fprintf(fp2,"%1.3f %1.3f 0 1 0 0\n", x1, x2);
            continue;
        }
        if((x1>=0 && x2>=0) || (x1<=0 && x2<=0)){
            fprintf(fp2,"%1.3f %1.3f 0 0 1 0\n", x1, x2); //check if it's in category 3
            continue;
        }else{
            fprintf(fp2,"%1.3f %1.3f 0 0 0 1\n", x1, x2); //check if it's in category 4
        }
    }
    fclose(fp1);
    fclose(fp2);     
}

void printDataset(){
    for (int i=0;i<4000;i++){
            printf("%1.3f %1.3f \n",datasetArray[i][0],datasetArray[i][1]);
        }
}

void setWeights(){
    int i,j;
    //Weights from the inputs to the first level
    for (i=0;i<H1;i++){
        for(j=0;j<d;j++){
            firstLevelWeights[i][j]=2*((float) rand()/RAND_MAX)-1;
        }
    }//first level bias
    for(j=0;j<H1;j++){
        firstLevelPolwsh[j]=2*((float) rand()/RAND_MAX)-1;
    }//second level weights
    for (i=0;i<H2;i++){
        for(j=0;j<H1;j++){
            secondLevelWeights[i][j]=2*((float) rand()/RAND_MAX)-1;
        } 
    }//second level bias
    for(j=0;j<H2;j++){
        secondLevelPolwsh[j]=2*((float) rand()/RAND_MAX)-1;
    }//third level weights
    for (i=0;i<H3;i++){
        for(j=0;j<H2;j++){
            thirdLevelWeightsNew[i][j]=2*((float) rand()/RAND_MAX)-1;
        } 
    }//third level bias
    for(j=0;j<H3;j++){
        thirdLevelPolwshNew[j]=2*((float) rand()/RAND_MAX)-1;
    }
    for (i=0;i<K;i++){
        for(j=0;j<H3;j++){
            thirdLevelWeights[i][j]=2*((float) rand()/RAND_MAX)-1;
        }
    }//exit level bias
    for(j=0;j<K;j++){
        thirdLevelPolwsh[j]=2*((float) rand()/RAND_MAX)-1;
    }
}

void forward_pass(float *x, int D, float *y, int k){
    //forward_pass method
    int i,j;
    for (i=0;i<H1;i++){//first level neurons
        firstLevelNeurons[i] = computeInerProduct(firstLevelWeights[i],x,D) + firstLevelPolwsh[i];//eswteriko ginomeno + polwsh.
        if (whatFunc==0){
            //according to the function that is defined above compute "g(u)".
            firstLevelNeuronsComplete[i]= relu(firstLevelNeurons[i]);
        }else{
            firstLevelNeuronsComplete[i]= tanh(firstLevelNeurons[i]);
        }
        
    }
    for (i=0;i<H2;i++){//second level neurons
        secondLevelNeurons[i]=computeInerProduct(secondLevelWeights[i],firstLevelNeuronsComplete,H1) + secondLevelPolwsh[i];
        if (whatFunc==0){
            secondLevelNeuronsComplete[i]= relu(secondLevelNeurons[i]);
        }else{
            secondLevelNeuronsComplete[i]= tanh(secondLevelNeurons[i]);
        }
    }
    for (i=0;i<H3;i++){//third level neurons
        thirdLevelNeuronsNew[i]=computeInerProduct(thirdLevelWeightsNew[i],secondLevelNeuronsComplete,H2) + thirdLevelPolwshNew[i];
        if (whatFunc==0){
            thirdLevelNeuronsCompleteNew[i]= relu(thirdLevelNeuronsNew[i]);
        }else{
            thirdLevelNeuronsCompleteNew[i]= tanh(thirdLevelNeuronsNew[i]);
        }
    }
    for (i=0;i<K;i++){//exit level
        thirdLevelNeurons[i]=computeInerProduct(thirdLevelWeights[i],thirdLevelNeuronsCompleteNew,H3) + thirdLevelPolwsh[i];
        y[i]= sigmoid(thirdLevelNeurons[i]);//sigmoid functions for exit level as our values are between 0 1
        
    }
}
void backprop(float *x, int D, float *t, int k){
    //backpropagation method
    int i,j;
    float fwpassResult[k];
    float sum;

    forward_pass(x,D,fwpassResult,k);
    //ypologise to sfalma "delta" se kathe epipedo
    //sfalma proorismou gia to teleftaio epipedo
    for (i=0;i<k;i++){
        deltaOut[i]=(fwpassResult[i]-t[i])*d_sigmoid(thirdLevelNeurons[i]);
    }
    //gia to trito krymmeno epipedo
    for (i=0;i<H3;i++){
        sum=0.0;
        for (j=0;j<k;j++){
            sum += thirdLevelWeights[j][i]*deltaOut[j];
        }if (whatFunc==0){
            deltaThirdLevel[i] = sum*d_relu(thirdLevelNeuronsNew[i]);
        }else{
            deltaThirdLevel[i] = sum*d_tanh(thirdLevelNeuronsNew[i]);
        }
    }

    //sfalma proorismou gia to deftero krymmeno epipedo
    for (i=0;i<H2;i++){
        sum=0.0;
        for (j=0;j<H3;j++){
            sum += thirdLevelWeightsNew[j][i]*deltaThirdLevel[j];
        }if (whatFunc==0){
            deltaSecondLevel[i] = sum*d_relu(secondLevelNeurons[i]);
        }else{
            deltaSecondLevel[i] = sum*d_tanh(secondLevelNeurons[i]);
        }
    }//sfalma proorismou gia to prwto krymmeno epipedo
    for (i=0;i<H1;i++){
        sum=0.0;
        for (j=0;j<H2;j++){
            sum += secondLevelWeights[j][i]*deltaSecondLevel[j];
        }if (whatFunc==0){
            deltaFirstLevel[i] = sum*d_relu(firstLevelNeurons[i]);
        }else{
            deltaFirstLevel[i] = sum*d_tanh(firstLevelNeurons[i]);
        }
    }
    //ypologise merikh paragwgo barous syndeshs
    for (i=0;i<H1;i++){
        for (j=0;j<d;j++){
            deh1[i][j]= deltaFirstLevel[i]*x[j];//gia epipedo H1
        }
        deh1polwsh[i]=deltaFirstLevel[i];
    }
    for (i=0;i<H2;i++){
        for(j=0;j<H1;j++){
            deh2[i][j] = deltaSecondLevel[i]*firstLevelNeuronsComplete[j];//gia to epipedo H2
        }
        deh2polwsh[i] = deltaSecondLevel[i];
    }
    for (i=0;i<H3;i++){
        for(j=0;j<H2;j++){
            deh3[i][j] = deltaThirdLevel[i]*secondLevelNeuronsComplete[j];//gia to epipedo H2
        }
        deh3polwsh[i] = deltaThirdLevel[i];
    }
    for (i=0;i<k;i++){
        for (j=0;j<H3;j++){
            deOut[i][j] = deltaOut[i]*thirdLevelNeuronsCompleteNew[j];//Gia thn eksodo
        }
        deOutPolwsh[i]=deltaOut[i];
    }
}
float computeInerProduct(float *x, float *y, int dmnsion){
    // compute iner product of two given vectors and return them.
    int i;
    float inerProduct=0.0;
    for (i=0;i<dmnsion;i++){
        inerProduct += x[i]*y[i];
    }
    return inerProduct;
}

float relu(float u){
    return fmaxf(0.0,u);
}
float d_relu(float u){
   if (u>0.0){
       return 1.0;
   }else{
       return 0.0;
   }
}
float tanhFunc(float u){
    return tanh(u);
}
float d_tanh(float u){
    return (1.0-tanh(u)*tanh(u));
}
float sigmoid(float u){
    return (1.0/(1.0+exp(-1*u)));
}
float d_sigmoid(float u){
	return (sigmoid(u)*(1.0 - sigmoid(u)));
}

void gradient_descent(){
    //ypologise tis metaboles mias epoxhs kai enhmerwse ta barh.
    int i,j,k,counter;
    int batchCounter=0;
    setZeros();
    //Ypologise thn synolikh metabolh(To athorisma twn merikwn) pou tha ypostei kathe neyrwnas sto telos mias epoxhs.
    if(B==trainValues){
        for (i=0;i<trainValues;i++){
            backprop(datasetArray[i],d,T[i],K);
            for (j=0;j<H1;j++){
                for (k=0;k<d;k++){
                    deh1Complete[j][k] += deh1[j][k];
                }
                deh1polwshComplete[j] += deh1polwsh[j];
            }
            for (j=0;j<H2;j++){
                for(k=0;k<H1;k++){
                    deh2Complete[j][k]+=deh2[j][k];
                }
                deh2polwshComplete[j]+=deh2polwsh[j];
            }
            for (j=0;j<H3;j++){
                for(k=0;k<H2;k++){
                    deh3Complete[j][k]+=deh3[j][k];
                }
                deh3polwshComplete[j]+=deh3polwsh[j];
            }
            for (j=0;j<K;j++){
                for (k=0;k<H3;k++){
                    deOutComplete[j][k]+=deOut[j][k];
                }
                deOutPolwshComplete[j]+= deOutPolwsh[j];
            }
        }
        setNewWeights();
    }else{
            for(i=0;i<trainValues;i++){
                backprop(datasetArray[i],d,T[i],K);
                for (j=0;j<H1;j++){
                    for (k=0;k<d;k++){
                        deh1Complete[j][k] += deh1[j][k];
                    }
                    deh1polwshComplete[j] += deh1polwsh[j];
                }
                for (j=0;j<H2;j++){
                    for(k=0;k<H1;k++){
                        deh2Complete[j][k]+=deh2[j][k];
                    }
                    deh2polwshComplete[j]+=deh2polwsh[j];
                }
                for (j=0;j<H3;j++){
                    for(k=0;k<H2;k++){
                        deh3Complete[j][k]+=deh3[j][k];
                    }
                    deh3polwshComplete[j]+=deh3polwsh[j];
                }
                for (j=0;j<K;j++){
                    for (k=0;k<H3;k++){
                        deOutComplete[j][k]+=deOut[j][k];
                    }
                    deOutPolwshComplete[j]+= deOutPolwsh[j];
                }
                batchCounter++;
                if(batchCounter==B){
                    setNewWeights();
                    setZeros();
                    batchCounter=0;
                }
            }
        }
}

int squareError(){
    //compute the square error training of an epoch
    int i,j;
    float error=0.0;
    for (i=0;i<trainValues;i++){
        forward_pass(datasetArray[i],d,Ycomplete[i],K);
        for(j=0;j<K;j++){
            error+=(T[i][j]-Ycomplete[i][j])*(T[i][j]-Ycomplete[i][j]);
        }
    }
    return (error/2);

}

void computeGeneralization(){
    int i,j,k,exitNeuron,answer=0;
    int pos =-3;
    float max =0.0;
    FILE *fp;
    for (i=0;i<testValues;i++){
        forward_pass(datasetArrayTest[i],d,Ytest[i],K);
        pos= -3;
        max=-10.0;
        exitNeuron=0;
        for(j=0;j<K;j++){
            if(Ytest[i][j]>max){
                max =Ytest[i][j];
                pos=j;
            }
        }
        for (k=0;k<K;k++){
            if(Ttest[i][k]==1){
                exitNeuron=k;
                break;
            }
        }
        if (pos==exitNeuron){
            fp=fopen("Positive2.txt","a+");
            for(k=0;k<d;k++){
                fprintf(fp,"%1.3f\t",datasetArrayTest[i][k]);
            }fprintf(fp,"%d\n",pos);
            fclose(fp);
            answer+=1;
        }else{
            fp=fopen("Negative2.txt","a+");
            for(k=0;k<d;k++){
                fprintf(fp,"%1.3f \t",datasetArrayTest[i][k]);
            }fprintf(fp,"%d\n",pos);
            fclose(fp);
            answer+=0;
        }
    }
    printf("\nFound correct : %d Generalization : %2.2f%%\n",answer,(float)100.0*answer/testValues);
}

void createArrays(){
    int i,j;
    FILE *fp;
    fp=fopen("DatasetTrain.txt","r");
    for (i=0;i<trainValues;i++){
        for(j=0;j<d;j++){
            fscanf(fp,"%f",&datasetArray[i][j]);
        }
        for(j=0;j<K;j++){
            fscanf(fp,"%f",&T[i][j]);
        }
    }
    fclose(fp);
    fp=fopen("DatasetTest.txt","r");
    for (i=0;i<testValues;i++){
        for(j=0;j<d;j++){
            fscanf(fp,"%f",&datasetArrayTest[i][j]);
        }
        for(j=0;j<K;j++){
            fscanf(fp,"%f",&Ttest[i][j]);
        }
    }
    fclose(fp);
}

void setNoise(){
    int i,j,position,newLabel;
    float chance=0.0;
    srand(time(NULL));
    for (i=0;i<trainValues;i++){
        for(j=0;j<K;j++){
            if(T[i][j]==1){
                position = j;
            }
        }
        chance=rand()%100;
        if (chance<=10){
            newLabel=rand()%4;
            while(newLabel==position){
                newLabel=rand()%4;
            }
            if (newLabel==1){
                T[i][0]=1;
                T[i][1]=0;
                T[i][2]=0;
                T[i][3]=0;
            }else if(newLabel==2){
                T[i][0]=0;
                T[i][1]=1;
                T[i][2]=0;
                T[i][3]=0;
            }else if(newLabel==3){
                T[i][0]=0;
                T[i][1]=0;
                T[i][2]=1;
                T[i][3]=0;
            }else if(newLabel==4){
                T[i][0]=0;
                T[i][1]=0;
                T[i][2]=0;
                T[i][3]=1;
            }
        }
    }
}

void setNewWeights(){
    int i,j,k;
    //bale tis synolikes metaboles sta brh
    for (i=0;i<H1;i++){
        for(j=0;j<d;j++){
            firstLevelWeights[i][j] -= (float)(h*deh1Complete[i][j]);
        }
    }
    for(j=0;j<H1;j++){
        firstLevelPolwsh[j] -= (float)(h*deh1polwshComplete[j]);
    }
    for (i=0;i<H2;i++){
        for(j=0;j<H1;j++){
            secondLevelWeights[i][j] -= (float)(h*deh2Complete[i][j]);
        } 
    }
    for(j=0;j<H2;j++){
        secondLevelPolwsh[j] -= (float)(h*deh2polwshComplete[j]);
    }
    for (i=0;i<H3;i++){
        for(j=0;j<H2;j++){
            thirdLevelWeightsNew[i][j] -= (float)(h*deh3Complete[i][j]);
        } 
    }
    for(j=0;j<H3;j++){
        thirdLevelPolwshNew[j] -= (float)(h*deh3polwshComplete[j]);
    }
    for (i=0;i<K;i++){
        for(j=0;j<H3;j++){
            thirdLevelWeights[i][j] -=  (float)(h*deOutComplete[i][j]);
        }
    }
    for(j=0;j<K;j++){
        thirdLevelPolwsh[j] -= (float)(h*deOutPolwshComplete[j]);
    }
}

void setZeros(){
    int i,j;
    //set every value to 0
    for(i=0;i<H1;i++){
        for(j=0;j<d;j++){
            deh1Complete[i][j]=0.0;
        }
        deh1polwshComplete[i]=0.0;
    }
    for(i=0;i<H2;i++){
        for(j=0;j<H1;j++){
            deh2Complete[i][j]=0.0;
        }
        deh2polwshComplete[i]=0.0;
    }
    for(i=0;i<H3;i++){
        for(j=0;j<H2;j++){
            deh3Complete[i][j]=0.0;
        }
        deh3polwshComplete[i]=0.0;
    }
    for(i=0;i<K;i++){
        for(j=0;j<H3;j++){
            deOutComplete[i][j]=0.0;
        }
        deOutPolwshComplete[i]=0.0;
    }
}