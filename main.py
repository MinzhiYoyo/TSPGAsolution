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
def render_map(max_node_num, need_regenerate=False):
    for the_node_num_is in range(10, max_node_num, 10):
        node_info = True if the_node_num_is <= 30 else False
        file_path = nodes_files[the_node_num_is] if not need_regenerate else None
        tsp = GA.TSPGA(the_node_num_is, file_path=file_path, save_path=nodes_files[the_node_num_is])
        tsp.render(title='generate nodes {}'.format(the_node_num_is), route=None, draw_notes=True, draw_costs=False,
                   draw_route=False,
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


# 画四条曲线
# 传入参数有：四条曲线的y值(y1, y2, y3, y4)，四条曲线的标签(label1, label2, label3, label4)
def plot_convergence_speed(**kwargs):
    plt.figure()
    y1 = kwargs['y1']
    y2 = kwargs['y2']
    y3 = kwargs['y3']
    y4 = kwargs['y4']
    # x为1到y的长度
    x = range(1, len(y1) + 1)
    plt.xlabel('Generation Number')
    plt.ylabel('Costs')
    # 一张图画四条曲线
    plt.plot(x, y1, label=kwargs['label1'], c='b', linewidth=1)
    plt.plot(x, y2, label=kwargs['label2'], c='g', linewidth=1)
    plt.plot(x, y3, label=kwargs['label3'], c='y', linewidth=1)
    plt.plot(x, y4, label=kwargs['label4'], c='c', linewidth=1)

    # 标注出每条曲线的y最小值，且x越小越好，对应位置画个点即可
    plt.scatter(x[y1.index(min(y1))], min(y1), c='r', marker='o', s=10)
    plt.scatter(x[y2.index(min(y2))], min(y2), c='r', marker='o', s=10)
    plt.scatter(x[y3.index(min(y3))], min(y3), c='r', marker='o', s=10)
    plt.scatter(x[y4.index(min(y4))], min(y4), c='r', marker='o', s=10)

    print('{:^2d}, ({:^2d}, {:^2.2f}), ({:^2d}, {:^2.2f}), ({:^2d}, {:^2.2f}), ({:^2d}, {:^2.2f})'.format(
        kwargs['node_num'], x[y1.index(min(y1))], min(y1), x[y2.index(min(y2))], min(y2), x[y3.index(min(y3))], min(y3), x[y4.index(min(y4))], min(y4)))

    plt.legend(loc='upper right')
    plt.savefig('./output/convergence_speed_{}.png'.format(int(kwargs['node_num'])))
    plt.show()


def plot_all_params(sleep_time=0):
    # 1. 两个都优化的情况，节点数量分别为10， 30， 50， 70， 90
    # node_nums = range(10, 51, 10)
    # for the_case_node in node_nums:
    #     plot_a_params(the_case_node, True, True)
    #     time.sleep(sleep_time)

    # 2. 节点数为10， 20， 30， 40， 50的情况
    #    每种情况都是四组实验
    # node_nums = range(10, 51, 10)
    # for the_case_node in node_nums:
    #     plot_a_params(the_case_node, True, True)
    #     time.sleep(sleep_time)
    #     plot_a_params(the_case_node, True, False)
    #     time.sleep(sleep_time)
    #     plot_a_params(the_case_node, False, True)
    #     time.sleep(sleep_time)
    #     plot_a_params(the_case_node, False, False)
    #     time.sleep(sleep_time)

    # 3. 节点收敛速度的对比
    node_nums = range(10, 51, 10)
    for the_case_node in node_nums:
        tspga = GA.TSPGA(the_case_node, file_path=nodes_files[the_case_node], save_path=nodes_files[the_case_node])
        # y1, y2, y3, y4分别代表：1.优化交叉，优化下一代，2.优化交叉，不优化下一代，3.不优化交叉，优化下一代，4.不优化交叉，不优化下一代
        dats = [
            get_log_data(the_case_node, True, True),
            get_log_data(the_case_node, True, False),
            get_log_data(the_case_node, False, True),
            get_log_data(the_case_node, False, False)
        ]
        y1 = [tspga.evaluate(route) for route in dats[0]]
        y2 = [tspga.evaluate(route) for route in dats[1]]
        y3 = [tspga.evaluate(route) for route in dats[2]]
        y4 = [tspga.evaluate(route) for route in dats[3]]
        plot_convergence_speed(y1=y1, y2=y2, y3=y3, y4=y4,
                               label1='Line1',  # 'Optimization Crossover, Optimization Next generation',
                               label2='Line2',  # 'Optimization Crossover, Not Optimization Next generation',
                               label3='Line3',  # 'Not Optimization Crossover, Optimization Next generation',
                               label4='Line4',  # 'Not Optimization Crossover, Not Optimization Next generation')
                               node_num=the_case_node)


def plot_a_params(node_num_: int, crossover: bool, next_generation: bool):
    the_case_routes = get_log_data(node_num_, crossover, next_generation)
    tsp = GA.TSPGA(node_num_, file_path=nodes_files[node_num_], save_path=nodes_files[node_num_])
    # for route in the_case_routes: tsp.render(title='{} nodes, {}, {}'.format(node_num, 'Optimization Crossover' if
    # crossover else 'Not Optimization Crossover', 'Optimization Next generation' if next_generation else 'Not
    # Optimization Next generation'), route=route, draw_notes=True, draw_costs=True, draw_route=True,
    # draw_node_info=False)

    tsp.render(title='{} nodes, {}, {}'.format(node_num_,
                                               'Optimization Crossover' if crossover else 'Not Optimization Crossover',
                                               'Optimization Next generation' if next_generation else 'Not Optimization Next generation'),
               route=the_case_routes[-1], draw_notes=True, draw_costs=False, draw_route=True, draw_node_info=False)
    plt.savefig('./output/nodes_{}_{}_{}.png'.format(node_num_, 'optimCross' if crossover else 'nooptimCross',
                                                     'optimNext' if next_generation else 'nooptimNext'))
    print('Save to {}, node is {}, costs is {}, route is {}'.format(
        './output/nodes_{}_{}_{}.png'.format(node_num_, 'optimCross' if crossover else 'nooptimCross',
                                             'optimNext' if next_generation else 'nooptimNext'),
        node_num_, tsp.evaluate(the_case_routes[-1]), the_case_routes[-1]))


def main():
    # print_distance_matrix(10)

    # 显示地图
    # render_map(max_node_num=51, need_regenerate=False)

    # run_all_params()

    # run_a_params(10, 2, True)
    # run_a_params(20, 2, True)
    # run_a_params(30, 2, True)
    # run_a_params(40, 2, True)
    # run_a_params(50, 2, True)

    plot_all_params(sleep_time=2)


if __name__ == '__main__':
    main()
