# Linux学习

[TOC]



## 1、系统目录结构

### 1.0 Linux的目录结构

> Linux系统的目录结构是一个树形结构
>
> Win系统有多个盘符，比如C盘，D盘，E盘等
>
> Linux没有盘符这个概念，只有一个根目录/，所有文件都在它下面

```
/usr/local/hello.txt
```

### 1.1 Linux路径描述方式

> - 在Linux系统中，路径之间的层级关系，使用：/来表示
> - 在Win系统中，路径之间的层级关系，使用：\来表示

### :book:练习

> 在根目录下有一个文件夹test，文件夹内有一个文件hello.txt，请描述文件的路径

```
/test/hello.txt
```

> 在根目录下有一个文件itheima.txt,请描述文件的路径

```
/itheima.txt
```

> 在根目录下有一个文件夹itcast，在itcast文件夹内有文件夹itheima,在itheima文件夹内有文件hello.txt,请描述文件的路径

```
/itcast/itheima/hello.txt
```

## 2、Linux命令基础

### 2.1 什么是命令、命令行

> - 命令行：即Linux终端(Terminal),是一种命令提示符页面。以纯“字符”的形式操作系统，可以使用各种字符化命令对系统发出操作指令。
> - 命令：即Linux程序。一个命令就是一个Linux程序。命令没有图形化页面，可以在命令行（终端中）提供字符化的反馈。

### 2.2 Linux命令基础格式

```
command [-options] [parameter]
```

> - command:命令本身
> - -options：[可选，非必填]命令的一些选项，可以通过选项控制命令的行为细节
> - parameter：[可选，非必填]命令的参数，多数用于命令的指向目标等
>
> 示例
>
> ```
> ls -l /home/itheima
> ```
>
> ls是命令本身，-l是选项，/home/itheima是参数
>
> - 意思是以列表的形式，显示/home/itheima目录内的内容
>
> ```
> cp -r test1 test2
> ```
>
> cp是命令本身，-r是选项，test1和test2是参数
>
> - 意思是复制文件夹test1成为test2