#include <iostream>
#include "load_data.h"
#include "ps.h"

namespace dmlc{
namespace linear{

class Worker : public ps::App{
    public:
        Worker(const char *train_file, const char *test_file) : 
                train_file_path(train_file), test_file_path(test_file){ }
        ~Worker(){
            delete train_data;
            //delete test_data;
        } 

        virtual void ProcessRequest(ps::Message* request){
	    //do nothing.
	    }

        float sigmoid(float x){
            if(x < -30) return 1e-6;
            else if(x > 30) return 1.0;
            else{
                double ex = pow(2.718281828, x);
                return ex / (1.0 + ex);
            }
        }

	    virtual bool Run(){
	        Process();
	    }
        void save_model(int st){
            char buffer[1024];
            snprintf(buffer, 1024, "%d", st);
            std::string filename = buffer;
            std::ofstream md;
            md.open("model/model_" + filename + ".txt");
            if(!md.is_open()) std::cout<<"open file error!"<<std::endl;
            //std::set<long int>::iterator iter;
            //for(iter = train_data->feaIdx.begin(); iter != train_data->feaIdx.end(); iter++){
            //    fea_all.push_back(*iter);
            //}
            std::cout<<"feaIdx size = "<<train_data->feaIdx.size()<<std::endl;
            //kv_.Wait(kv_.Pull(fea_all, &w_all));
            kv_.Wait(kv_.Pull(init_index, &w_all));
            for(int i = 0; i < fea_all.size(); i++){
                    md << fea_all[i]<<"\t"<<w_all[i]<<std::endl;
            }
            md.close();
        }
        
        void predict(int st){
           char buffer[1024];
           snprintf(buffer, 1024, "%d", st);
           std::string filename = buffer;
           std::ofstream md;
           md.open("pred_" + filename + ".txt");
           if(!md.is_open()) std::cout<<"open pred file failure!"<<std::endl;
           std::cout<<"test_data size = "<<test_data->fea_matrix.size()<<std::endl;
           for(int i = 0; i < test_data->fea_matrix.size(); i++) {
               float x = 0.0;
               for(int j = 0; j < test_data->fea_matrix[i].size(); j++) {
                   long index = test_data->fea_matrix[i][j].idx;
                   int value = test_data->fea_matrix[i][j].val;
                   x += w_all[index] * value;
               }
               double pctr;
               if(x < -30){
                       pctr = 1e-6;
               }
               else if(x > 30){
                       pctr = 1.0;
               }
               else{
                       double ex = pow(2.718281828, x);
                       pctr = ex / (1.0 + ex);
               }

               md<<pctr<<"\t"<<1 - test_data->label[i]<<"\t"<<test_data->label[i]<<std::endl;
           }
           md.close();
        }

        void batch_gradient_calculate(int &row){
            int index = 0; float value = 0.0; float pctr = 0;
            for(int line = 0; line < batch_size; line++){
                if(row >= train_data->fea_matrix.size()) break;
                std::vector<float> g;
                std::vector<ps::Key> keys;
                std::vector<float> values;
                std::vector<float> w;
                for(int j = 0; j < train_data->fea_matrix[row].size(); j++){//for one instance
                    index = train_data->fea_matrix[row][j].idx;
                    keys.push_back(index);
                    value = train_data->fea_matrix[row][j].val;
                    values.push_back(value);
                }
                kv_.Wait(kv_.Pull(keys, &w));
                float wx = bias;
                for(int j = 0; j < w.size(); j++){
                    wx += w[j] * values[j];
                }
                pctr = sigmoid(wx);
                g.resize(keys.size());
                float delta = pctr - train_data->label[row];
                for(int j = 0; j < keys.size(); j++){
                    g[j] = delta * values[j];
                }
                kv_.Wait(kv_.Push(keys, g));
                row++;
            }
        }

        virtual void Process(){
	        rank = ps::MyRank();
            snprintf(train_data_path, 1024, "%s-%05d", train_file_path, rank);
            train_data = new Load_Data(train_data_path);
            train_data->load_all_data();
            std::cout<<"rank = "<<rank<<" fea_matrix size = "<<train_data->fea_matrix.size()<<std::endl;

            //std::vector<ps::Key> init_index;
            init_index.clear();
            for(int i = 0; i < 1e8; i++){
                init_index.push_back(i);
            }
            std::vector<float> init_val(1e8, 0.0);
            kv_.Wait(kv_.Push(init_index, init_val));

            for(int i = 0; i < step; i++){
                int row = 0;
                while(row < train_data->fea_matrix.size()){
                    batch_gradient_calculate(row);
                    if(row % 20000 == 0) std::cout<<"row = "<<row<<std::endl;
                }//end for minibatch
            }//end for
            if(rank == 0){
                std::cout<<"end"<<std::endl;
                save_model(step);
                snprintf(test_data_path, 1024, "%s-%05d", test_file_path, rank);
                std::cout<<" test data_path======================"<<test_data_path<<std::endl;
                test_data = new Load_Data(test_data_path);
                test_data->load_all_data();

                predict(step);
            }
        }//end process

    public:
        std::vector<ps::Key> init_index;
        std::vector<ps::Key> fea_all;
        std::vector<float> w_all;	
        Load_Data *train_data;
        Load_Data *test_data;
        const char *train_file_path;
        const char *test_file_path;
        char train_data_path[1024];
        char test_data_path[1024];
        int batch_size = 300;
        int rank;
        float bias = 0.0;
        float alpha = 0.1;
        float beta = 1.0;
        float lambda1 = 0.001;
        float lambda2 = 0.0;
        int step = 2;
        ps::KVWorker<float> kv_;
};//end class worker

}//end namespace linear
}//end namespace dmlc 
