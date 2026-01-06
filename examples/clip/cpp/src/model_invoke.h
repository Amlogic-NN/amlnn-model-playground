#ifndef MODEL_INVOKE_H
#define MODEL_INVOKE_H

#include <string>
#include <vector>
#include <map>

void* init_network_file(const char *model_path);
std::vector<std::string> process_image_dir(void *context_model, const std::string& json_path, const std::string& base_dir = "", const std::string& json_filename = "");
int destroy_network(void *qcontext);

#endif // MODEL_INVOKE_H

