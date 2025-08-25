fn main() {
    // MAXIMUM PERFORMANCE BUILD SCRIPT
    // Ignore compile time and binary size - optimize for runtime speed only
    
    println!("cargo:rustc-env=TARGET_CPU=native");
    
    // Aggressive linker optimizations
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-arg=-Wl,--gc-sections");
        println!("cargo:rustc-link-arg=-Wl,--strip-all");
        println!("cargo:rustc-link-arg=-Wl,-O3");
        println!("cargo:rustc-link-arg=-Wl,--as-needed");
    }
    
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-arg=-Wl,-dead_strip");
        println!("cargo:rustc-link-arg=-Wl,-O3");
    }
    
    // Enable ALL available CPU features for maximum performance
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // Modern CPU features
        if std::arch::is_x86_feature_detected!("avx2") {
            println!("cargo:rustc-cfg=feature=\"avx2\"");
        }
        if std::arch::is_x86_feature_detected!("fma") {
            println!("cargo:rustc-cfg=feature=\"fma\"");
        }
        if std::arch::is_x86_feature_detected!("sse4.2") {
            println!("cargo:rustc-cfg=feature=\"sse4_2\"");
        }
        if std::arch::is_x86_feature_detected!("sse4.1") {
            println!("cargo:rustc-cfg=feature=\"sse4_1\"");
        }
        if std::arch::is_x86_feature_detected!("ssse3") {
            println!("cargo:rustc-cfg=feature=\"ssse3\"");
        }
        if std::arch::is_x86_feature_detected!("sse3") {
            println!("cargo:rustc-cfg=feature=\"sse3\"");
        }
        if std::arch::is_x86_feature_detected!("popcnt") {
            println!("cargo:rustc-cfg=feature=\"popcnt\"");
        }
        if std::arch::is_x86_feature_detected!("bmi1") {
            println!("cargo:rustc-cfg=feature=\"bmi1\"");
        }
        if std::arch::is_x86_feature_detected!("bmi2") {
            println!("cargo:rustc-cfg=feature=\"bmi2\"");
        }
        if std::arch::is_x86_feature_detected!("lzcnt") {
            println!("cargo:rustc-cfg=feature=\"lzcnt\"");
        }
        
        // Newer features (if available)
        if std::arch::is_x86_feature_detected!("avx512f") {
            println!("cargo:rustc-cfg=feature=\"avx512f\"");
        }
    }
    
    // Force maximum optimization flags
    println!("cargo:rustc-env=OPT_LEVEL=3");
    println!("cargo:rustc-env=CODEGEN_UNITS=1");
    
    println!("cargo:rerun-if-changed=build.rs");
}