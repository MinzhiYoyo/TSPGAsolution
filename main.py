import os.path
import time

import matplotlib.pyplot as plt
import numpy as np

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
    for the_node_num_is in range(10, 11, 10):
        node_info = True if the_node_num_is <= 30 else False
        tsp = GA.TSPGA(the_node_num_is, file_path=nodes_files[the_node_num_is], save_path=nodes_files[the_node_num_is])
        tsp.render(title='generate nodes', route=None, draw_notes=True, draw_costs=False, draw_route=False,
                   draw_node_info=node_info)
        time.sleep(2)
        plt.savefig('./output/generate_nodes_{}.png'.format(the_node_num_is))


# 对比各种参数下的表现
def run_all_params():
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
    nodes_ = range(10, 51, 10)
    for n in nodes_:
        for i in range(params['times']):
            run_a_params(n, params['crossover'][i], params['next_generation'][i])


def run_a_params(node_num_: int, crossover: int, next_generation: bool):
    rst_output = './log/routes_{}_{}_{}.npy'.format(
        node_num_,
        'optimCross' if crossover == 2 else 'nooptimCross',
        'optimNext' if next_generation else 'nooptimNext'
    )

    tspga = GA.GA(generation_num, population_num, mutation_rate, crossover_rate, node_num_,
                  file_path=nodes_files[node_num_],
                  save_path=nodes_files[node_num_])
    tspga.simulate(crossover_num=crossover, separate=next_generation,
                   convergence_exit=None, save_path=rst_output, show=True)


def print_distance_matrix(node_num_is):
    tsp = GA.TSPGA(node_num_is, file_path=nodes_files[node_num_is],
                   save_path=nodes_files[node_num_is])
    # 把tsp.distance_table打印出来
    for row in tsp.distance_table:
        for element in row:
            print('{:.2f}'.format(element), end=' ')
        print('')


def get_log_data(the_case_node_num: int, the_case_opti_cross: bool, the_case_opti_next: bool):
    the_case_opti_cross = 'optimCross' if the_case_opti_cross else 'nooptimCross'
    the_case_opti_next = 'optimNext' if the_case_opti_next else 'nooptimNext'
    the_case_routes = None
    log_files = os.listdir('./log/')
    for log_file in log_files:
        if log_file.endswith('.npy'):
            file_info_list = log_file.split('.')[0].split('_')
            if ('routes' == file_info_list[0] and int(file_info_list[-3]) == the_case_node_num and
                    file_info_list[-2] == the_case_opti_cross and file_info_list[-1] == the_case_opti_next):
                log_file_path = os.path.join('./log/', log_file)
                the_case_routes = np.load(log_file_path)
    return the_case_routes


def plot_all_params(sleep_time=0):
    # 1. 两个都优化的情况，节点数量分别为10， 30， 50， 70， 90
    node_nums = [10, 30, 50, 70, 90]
    for the_case_node in node_nums:
        plot_a_params(the_case_node, True, True)
        time.sleep(sleep_time)


def plot_a_params(node_num_: int, crossover: bool, next_generation: bool):
    the_case_routes = get_log_data(node_num_, crossover, next_generation)
    tsp = GA.TSPGA(node_num_, file_path=nodes_files[node_num_], save_path=nodes_files[node_num_])
    # for route in the_case_routes:
    #     tsp.render(title='{} nodes, {}, {}'.format(node_num,
    #                                                'Optimization Crossover' if crossover else 'Not Optimization Crossover',
    #                                                'Optimization Next generation' if next_generation else 'Not Optimization Next generation'),
    #                route=route, draw_notes=True, draw_costs=True, draw_route=True, draw_node_info=False)

    tsp.render(title='{} nodes, {}, {}'.format(node_num_,
                                               'Optimization Crossover' if crossover else 'Not Optimization Crossover',
                                               'Optimization Next generation' if next_generation else 'Not Optimization Next generation'),
               route=the_case_routes[-1], draw_notes=True, draw_costs=True, draw_route=True, draw_node_info=False)
    plt.savefig('./output/nodes_{}_{}_{}.png'.format(node_num_, 'optimCross' if crossover else 'nooptimCross',
                                                     'optimNext' if next_generation else 'nooptimNext'))
    print('Save to {}, node is {}, costs is {}, route is {}'.format(
        './output/nodes_{}_{}_{}.png'.format(node_num_, 'optimCross' if crossover else 'nooptimCross',
                                             'optimNext' if next_generation else 'nooptimNext'),
        node_num_, tsp.evaluate(the_case_routes[-1]), the_case_routes[-1]))


def main():
    # print_distance_matrix(10)
    # render_map()
    run_all_params()

    # run_a_params(10, 2, True)
    # run_a_params(30, 2, True)
    # run_a_params(50, 2, True)
    # run_a_params(70, 2, True)
    # run_a_params(90, 2, True)

    # plot_all_params(sleep_time=3)


if __name__ == '__main__':
    main()
