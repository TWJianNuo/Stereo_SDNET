#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> bnmorph_find_coorespond_pts_cuda(
    torch::Tensor binMapsrc,
    torch::Tensor binMapdst,
    torch::Tensor xx,
    torch::Tensor yy,
    torch::Tensor sxx,
    torch::Tensor syy,
    torch::Tensor cxx,
    torch::Tensor cyy,
    float pixel_distance_weight,
    float alpha_distance_weight,
    float pixel_mulline_distance_weight,
    float alpha_padding
    );

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

//torch::Tensor bnmorph_find_coorespond_pts(
std::vector<torch::Tensor> bnmorph_find_coorespond_pts(
    torch::Tensor binMapsrc,
    torch::Tensor binMapdst,
    torch::Tensor xx,
    torch::Tensor yy,
    torch::Tensor sxx,
    torch::Tensor syy,
    torch::Tensor cxx,
    torch::Tensor cyy,
    float pixel_distance_weight,
    float alpha_distance_weight,
    float pixel_mulline_distance_weight,
    float alpha_padding
    ) {
    CHECK_INPUT(binMapsrc)
    CHECK_INPUT(binMapdst)
    CHECK_INPUT(xx)
    CHECK_INPUT(yy)
    CHECK_INPUT(sxx)
    CHECK_INPUT(syy)
    CHECK_INPUT(cxx)
    CHECK_INPUT(cyy)
    std::vector<torch::Tensor> results_bindings = bnmorph_find_coorespond_pts_cuda(binMapsrc, binMapdst, xx, yy, sxx, syy, cxx, cyy, pixel_distance_weight, alpha_distance_weight, pixel_mulline_distance_weight, alpha_padding);


//    const int width = binMapsrc.size(3);
//    const int height = binMapsrc.size(2);
//    const int batchSize = binMapsrc.size(0);
//    std:: cout << "width: " << width << ", height: " << height << std::endl;

//    auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided);
//    auto orgx_itor = results_bindings[0].accessor<float,4>();
//    auto validPixel = torch::ones({batchSize, 1, height, width}, options);
//    auto validPixel_itor = validPixel.accessor<float,4>();

//
//    validPixel_itor[0][0][200][800] = 0.0;
//    validPixel_itor[0][0][1][1] = 0.0;
//    int ckx = 0;
//    int cky = 0;
//    for(int t = 0; t < batchSize; t++){
//        for(int i = 0; i < height; i++){
//            for(int j = 0; j < width; j++){
//                if (orgx_itor[t][0][i][j] > -1e-3){
//                    if(validPixel_itor[t][0][i][j] > 1e-3){
//                        for(int m = -sparsityRad; m < sparsityRad + 1; m++){
//                            for(int n = -sparsityRad; n < sparsityRad + 1; n++){
//                                ckx = m + i;
//                                cky = n + j;
//                                if((ckx >= 0) && (cky >=0) && (ckx < width) && (cky < height)){
//                                     validPixel_itor[0][0][40][600] = 0.0;
//                                }
//                            }
//                        }
//                    }
//                }
//            }
//        }
//    }
//    validPixel = validPixel.to(torch::kCUDA);


//    torch::Tensor foo = torch::ones({12, 12,12}, options);
//    torch::Tensor foo2 = torch::ones_like(binMapdst);
//    auto foo_a = foo.accessor<float,3>();
//    auto foo_b = foo2.accessor<float,4>();
//    float trace = 0;
//
//
//    auto test = binMapsrc.accessor<float,4>();
//    auto x_org = results_bindings[0].accessor<float,4>();
//    auto y_org = results_bindings[1].accessor<float,4>();
//    auto x_dst = results_bindings[2].accessor<float,4>();
//    auto y_dst = results_bindings[3].accessor<float,4>();
//    auto indicator_map = torch::ones({binMapsrc.size(0), binMapsrc.size(1), binMapsrc.size(2), binMapsrc.size(3)});
//    auto indicator_map_itor = indicator_map.accessor<float,4>();
//    for(int t = 0; t < batchSize; t++){
//        for(int i = 0; i < height; i++){
//            for(int j = 0; j < width; j++){
//                if(x_org[t][0][i][j] > -1e-3){
//                    if(indicator_map_itor[t][0][i][j] > 1e-5){
//                    }
//                    else{
//                    }
//                }
//            }
//        }
//    }
    return results_bindings;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("find_corespond_pts", &bnmorph_find_coorespond_pts, "find corresponding points of two binary map, under certain search range and allowed distance");
}
