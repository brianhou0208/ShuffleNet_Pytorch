from calflops import calculate_flops

def flops_params(model, input_shape=(1, 3, 224, 224)):
    flops, macs, params = calculate_flops(
        model=model, 
        input_shape=input_shape,
        output_as_string=True,
        print_detailed=False,
        print_results=False,
        output_precision=4
    )
    print("| FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))