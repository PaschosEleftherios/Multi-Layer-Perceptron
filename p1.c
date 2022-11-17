#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#define trainValues 4000
#define testValues 4000
#define d 2 //arithmos eisodwn
#define K 4 //arithmos kathgoriwn
#define H1 12 //arithmos neyrwnwn prwto krymmeno epipedo
#define H2 12 //arithmos neyrwnwn deftero krymmeno epipedo
#define h 0.0005 //rythmos mathishs
#define whatFunc 1 //What function gia ton kathorismo ths relu an ==0 h tanh an ==1
#define errorDiff 10 //katwflh termatismou metaksi 2 epoxwn
#define B 4000

//Ylopoioun to Dataset S1.
void createDataset();
void printDataset();
void createArrays();
void setNoise();

void setWeights();//Dhnei mia fora sthn arxh tyxaies times sta barh
void forward_pass(float *x, int D, float *y, int k);//eythi perasma
void backprop(float *x, int D, float *t, int k);//opisthodromish
float computeInerProduct(float *x, float *y, int dmnsion);//ypologismos eswterikou ginomenou


void gradient_descent();
void setNewWeights();
void setZeros();
void computeGeneralization();//ypologizei thn genikeysh
int squareError();//ypologizei to tetragwniko sfalma


//Synarthseis pou xrhsimopoiountai prokeimenou
//na anaparasthsoume thn relu/tanh/logistikh
//kathws kai tis paragwgous afton
float relu(float u);
float d_relu(float u);
float tanhFunc(float u);
float d_tanh(float u);
float sigmoid(float u);
float d_sigmoid(float u);


//weights and biases
float firstLevelWeights[H1][d];
float secondLevelWeights[H2][H1];
float thirdLevelWeights[K][H2];
float firstLevelPolwsh[H1];
float secondLevelPolwsh[H2];
float thirdLevelPolwsh[K];


//Arrays for the training and testing of the network
float datasetArray[trainValues][d];
float T[trainValues][K];
float datasetArrayTest[testValues][d];//gia ton elegxo
float Ttest[testValues][K];//For testing the network will be used in Generalization
float Ycomplete[trainValues][K];//gia to  sfalma
float Ytest[testValues][K];//For testing the network will be used in Generalization

//Arrays for reflecting the Neurons of the network 
//and the neurons after their activation fuction(...*Complete).
float firstLevelNeurons[H1];
float firstLevelNeuronsComplete[H1];
float secondLevelNeurons[H2];
float secondLevelNeuronsComplete[H2];
float thirdLevelNeurons[K];
float thirdLevelNeuronsComplete[K];

//sfalma proorismou gia kathe epipedo
float deltaOut[K];
float deltaSecondLevel[H2];
float deltaFirstLevel[H1];

//merikh metabolh kathe epipedou
float deh1[H1][d];
float deh1polwsh[H1];
float deh2[H2][H1];
float deh2polwsh[H2];
float deOut[K][H2];
float deOutPolwsh[K];

//synolikos rythmos metabolhs
float deh1Complete[H1][d];
float deh1polwshComplete[H1];
float deh2Complete[H2][H1];
float deh2polwshComplete[H2];
float deOutComplete[K][H2];
float deOutPolwshComplete[K];


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
            if (abs(sfalma-sfalmaBefore)<=errorDiff && sfalma<=400){
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
    }//exit level weights
    for (i=0;i<K;i++){
        for(j=0;j<H2;j++){
            thirdLevelWeights[i][j]=2*((float) rand()/RAND_MAX)-1;
        }
    }//exit level bias
    for(j=0;j<K;j++){
        thirdLevelPolwsh[j]=2*((float) rand()/RAND_MAX)-1;
    }
}

void forward_pass(float *x, int D, float *y, int k){
    //bres to dianisma y diastash k gia to dothen protypo x diastashs D.
    int i,j;
    for (i=0;i<H1;i++){//first level neurons
        firstLevelNeurons[i] = computeInerProduct(firstLevelWeights[i],x,D) + firstLevelPolwsh[i];//eswteriko ginomeno + polwsh.
        if (whatFunc==0){
            //analoga ti synarthsh exei dhlwthei sto define gia ta krymmena epipeda ypologizei to "g(u)".
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
    for (i=0;i<K;i++){//exit level
        thirdLevelNeurons[i]=computeInerProduct(thirdLevelWeights[i],secondLevelNeuronsComplete,H2) + thirdLevelPolwsh[i];
        y[i]= sigmoid(thirdLevelNeurons[i]);//sto epipedo eksodou logistikh synarthsh kathw theloume times anamesa se 0-1.
        
    }
}
void backprop(float *x, int D, float *t, int k){
    //opisthodromish
    int i,j;
    float fwpassResult[k];
    float sum;

    forward_pass(x,D,fwpassResult,k);
    //ypologise to sfalma "delta" se kathe epipedo
    //sfalma proorismou gia to teleftaio epipedo
    for (i=0;i<k;i++){
        deltaOut[i]=(fwpassResult[i]-t[i])*d_sigmoid(thirdLevelNeurons[i]);
    }
    //sfalma proorismou gia to deftero krymmeno epipedo
    for (i=0;i<H2;i++){
        sum=0.0;
        for (j=0;j<k;j++){
            sum += thirdLevelWeights[j][i]*deltaOut[j];
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
    for (i=0;i<k;i++){
        for (j=0;j<H2;j++){
            deOut[i][j] = deltaOut[i]*secondLevelNeuronsComplete[j];//Gia thn eksodo
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
    //Ypologise ton synoliko rythmo metabolhs(To athorisma twn merikwn) pou tha ypostei kathe neyrwnas sto telos mias epoxhs.
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
            for (j=0;j<K;j++){
                for (k=0;k<H2;k++){
                    deOutComplete[j][k]+=deOut[j][k];
                }
                deOutPolwshComplete[j]+= deOutPolwsh[j];
            }
        }
        setNewWeights();
    }else{
        counter=0;
        while(counter!=trainValues){
            for(i=batchCounter;i<B;i++){
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
                for (j=0;j<K;j++){
                    for (k=0;k<H2;k++){
                        deOutComplete[j][k]+=deOut[j][k];
                    }
                    deOutPolwshComplete[j]+= deOutPolwsh[j];
                }
            }
            setNewWeights();
            setZeros();
            batchCounter+=B;
            counter+=B;
        }
    }
}

int squareError(){
    //ypologise to  sfalma ekpaideyshs
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
        pos= -10;
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
            fp=fopen("Positive.txt","a+");
            for(k=0;k<d;k++){
                fprintf(fp,"%1.3f \t",datasetArrayTest[i][k]);
            }
            fprintf(fp,"%d\n",pos);
            fclose(fp);
            answer+=1;
        }else{
            fp=fopen("Negative.txt","a+");
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
        firstLevelPolwsh[i] -= (float)(h*deh1polwshComplete[i]);
    }
    for (i=0;i<H2;i++){
        for(j=0;j<H1;j++){
            secondLevelWeights[i][j] -= (float)(h*deh2Complete[i][j]);
        }
        secondLevelPolwsh[i] -= (float)(h*deh2polwshComplete[i]); 
    }
    for (i=0;i<K;i++){
        for(j=0;j<H2;j++){
            thirdLevelWeights[i][j] -=  (float)(h*deOutComplete[i][j]);
        }
        thirdLevelPolwsh[i] -= (float)(h*deOutPolwshComplete[i]);
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
    for(i=0;i<K;i++){
        for(j=0;j<H2;j++){
            deOutComplete[i][j]=0.0;
        }
        deOutPolwshComplete[i]=0.0;
    }
}
