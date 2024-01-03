import utils

if __name__ == '__main__':
    model_id = "qwen/Qwen-1_8B-Chat"
    utils.modelscope_download(model_id)

    # model_name_list = [
    #     "hfl/chinese-pert-large-mrc",
    #     # "wptoux/albert-chinese-large-qa",
    # ]
    # utils.model_download(model_name_list)
