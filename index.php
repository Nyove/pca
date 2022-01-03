<?php
    header("Content-Type:text/html;charset=指定编码");
    // 保存图片
    move_uploaded_file($_FILES["testImg"]["tmp_name"], "test.png");

    // 模拟cmd执行python脚本
    exec("python pca.py 2>&1", $output, $state);
    print_r($output);   # 打印：命令行错误信息
    echo "<br>";    # 换行
    echo "state: $state";   # 打印：0-执行成功，1-失败

    // 页面效果
    echo "<img src='test.png'><br>";
    echo $output[5];
?>
