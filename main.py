import os.path
import matplotlib.pyplot as plt
import GA


def get_time_info():
    import time
    return time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))


nodes_files = {
    10: 'data/nodes_10.npy',
    20: 'data/nodes_20.npy',
    30: 'data/nodes_30.npy',
    40: 'data/nodes_40.npy',
    50: 'data/nodes_50.npy',
    60: 'data/nodes_60.npy',
    70: 'data/nodes_70.npy',
    80: 'data/nodes_80.npy',
    90: 'data/nodes_90.npy',
    100: 'data/nodes_100.npy'
}

# 定义超参数
population_num = 1000
generation_num = 500
mutation_rate = 0.3
crossover_rate = 0.8
node_num = 20


# 生成地图示意图
def render_map():
    the_node_num_is = 20
    tsp = GA.TSPGA(the_node_num_is, file_path=nodes_files[the_node_num_is], save_path=nodes_files[the_node_num_is])
    tsp.render(title='generate nodes', route=None, draw_notes=True, draw_costs=False, draw_route=False,
               draw_node_info=True)
    plt.show()


# 对比各种参数下的表现
def compare_params():
    # 1. 是否优化交叉方法
    # 2. 是否优化下一代生成算法
    params = {
        'times': 4,
        'crossover': [2, 2, 1, 1],
        'next_generation': [True, False, True, False],
        'info': [
            '优化交叉方法，优化下一代生成算法',
            '优化交叉方法，不优化下一代生成算法',
            '不优化交叉方法，优化下一代生成算法',
            '不优化交叉方法，不优化下一代生成算法'
        ]
    }
    nodes_ = [10, 30, 50, 70, 90]
    for n in nodes_:
        for i in range(params['times']):
            rst_output = './log/log_{}_{}_{}_{}.npy'.format(
                get_time_info(), n,
                'optimCross' if params['crossover'][i] == 2 else 'nooptimCross',
                'optimNext' if params['next_generation'][i] else 'nooptimNext'
            )

            tspga = GA.GA(generation_num, population_num, mutation_rate, crossover_rate, n, file_path=nodes_files[n],
                          save_path=nodes_files[n])
            tspga.simulate(crossover_num=params['crossover'][i], separate=params['next_generation'][i],
                           convergence_exit=None, save_path=rst_output, show=True)


def main():
    # render_map()
    compare_params()


if __name__ == '__main__':
    main()
