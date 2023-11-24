from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import *
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import InequalityToEquality, LinearEqualityToPenalty
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import SPSA
from qiskit import Aer
import pandas as pd
import numpy as np
import uuid


# Дополнительные функции
def get_day_time_combinations():
    # Возвращает все возможные комбинации дня недели и времени
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    times = ['9:00-10:00', '10:00-11:00', '11:00-12:00', '12:00-13:00', '13:00-14:00', '14:00-15:00', '15:00-16:00',
             '16:00-17:00']
    return [(day, time) for day in days for time in times]


def get_teacher_subject_mapping():
    # Возвращает отображение преподаватель-предмет
    return {
        'Иванов': ['Квантовая механика', 'Квантовая теория информации'],
        'Петров': ['Квантовые вычисления', 'Сложность квантовых алгоритмов'],
        'Сидоров': ['Квантовые алгоритмы в логистике', 'Квантовое машинное обучение'],
        'Карпов': ['Моделирование квантовых систем', 'Квантовые алгоритмы в химии'],
        'Соколов': ['Физическая реализация квантовых компьютеров', 'Моделирование квантовых алгоритмов']
    }


def calculate_conflicts(result, qp, day_time_combinations, teachers_subjects):
    conflicts = 0
    for day, time in day_time_combinations:
        for group in ['Группа1', 'Группа2']:
            # Проверка нарушения ограничения по количеству лекций в день для группы
            lectures_count = sum(result.x['_'.join([teacher, subject, group, day, time])]
                                 for teacher, subjects in teachers_subjects.items()
                                 for subject in subjects)
            if lectures_count > 6:
                conflicts += 1

            # Проверка нарушения ограничения по количеству занятий по одному предмету в день для группы
            for subject in set(subject for subjects in teachers_subjects.values() for subject in subjects):
                subject_lectures_count = sum(result.x['_'.join([teacher, subject, group, day, time])]
                                             for teacher, subjects in teachers_subjects.items()
                                             if subject in subjects)
                if subject_lectures_count > 2:
                    conflicts += 1

            # Проверка нарушения ограничения неработы преподавателей в определенные дни
            for teacher, subjects in teachers_subjects.items():
                if teacher == 'Иванов' and day == 'Wednesday':
                    for subject in subjects:
                        if result.x['_'.join([teacher, subject, group, day, time])] == 1:
                            conflicts += 1

    return conflicts


# Определение целевой функции
def objective_function(result, qp, day_time_combinations, teachers_subjects):
    conflicts = calculate_conflicts(result, qp, day_time_combinations, teachers_subjects)
    return conflicts


# Определение целевой функции
def your_objective_function(result, qp, day_time_combinations, teachers_subjects):
    conflicts = calculate_conflicts(result, qp, day_time_combinations, teachers_subjects)
    return conflicts


# Основная функция для создания расписания
def create_schedule():
    # Получение данных
    teachers_subjects = get_teacher_subject_mapping()
    day_time_combinations = get_day_time_combinations()

    # Создание списка переменных
    variables = []

    # Создание квадратичной программы
    qp = QuadraticProgram()

    # Создание переменных в квадратичной программе
    variables = {}
    for day, time in day_time_combinations:
        for teacher, subjects in teachers_subjects.items():
            for subject in subjects:
                for group in ['Группа1', 'Группа2']:
                    variable_name = f"{teacher}_{subject}_{group}_{day}_{time}"
                    variables[variable_name] = qp.binary_var(name=variable_name)

    # Определение ограничений
    for day, time in day_time_combinations:
        for group in ['Группа1', 'Группа2']:
            # Ограничение на количество лекций в день для группы
            constraint_name_day_lecture_limit = f"day_lecture_limit_{group}_{day}_{uuid.uuid4().hex}"
            qp.linear_constraint(linear={var: 1 for var in variables
                                         if group in var and day in var},
                                 sense='<=',
                                 rhs=6,
                                 name=constraint_name_day_lecture_limit)

            # Ограничение на количество занятий по одному предмету в день для группы
            for subject in set(subject for subjects in teachers_subjects.values() for subject in subjects):
                constraint_name_subject_limit = f"subject_limit_{subject}_{group}_{day}_{uuid.uuid4().hex}"
                qp.linear_constraint(linear={var: 1 for var in variables
                                             if group in var and day in var and subject in var},
                                     sense='<=',
                                     rhs=2,
                                     name=constraint_name_subject_limit)

                # Ограничение на неработу преподавателей в определенные дни
                for teacher, subjects in teachers_subjects.items():
                    if teacher == 'Иванов' and day == 'Wednesday':
                        constraint_name_unavailability = f"unavailability_{teacher}_{subject}_{group}_{day}_{uuid.uuid4().hex}"
                        qp.linear_constraint(linear={var: 1 for var in variables
                                                     if group in var and day in var and teacher in var},
                                             sense='==',
                                             rhs=0,
                                             name=constraint_name_unavailability)

    # Решение задачи оптимизации
    optimizer = CplexOptimizer()
    result = optimizer.solve(qp)

    # Получение результата
    data = []
    for var in variables:
        if var in result.x and result.x[var] == 1:
            teacher, subject, group, day, time = var.split('_')
            data.append([teacher, subject, group, day, time])

    # Отображение результатов в виде таблицы
    display_schedule(data)


# Функция для отображения расписания в виде таблицы
def display_schedule(data):
    # Создание таблицы
    df = pd.DataFrame(data, columns=['Преподаватель', 'Предмет', 'Группа', 'День', 'Час'])

    # Вывод таблицы
    print(df)


# Вызов основной функции
create_schedule()
