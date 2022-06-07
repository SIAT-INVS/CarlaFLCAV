

def generate_imageset_txt(filepath, start_frame, stop_frame, step=1):
    with open(filepath, 'w') as file:
        for i in range(start_frame, stop_frame, step):
            file.writelines("{:0>6d}\n".format(i))


def main():
    generate_imageset_txt('/tmp/training.txt', 250, 4750, step=3)


if __name__ == "__main__":
    main()
