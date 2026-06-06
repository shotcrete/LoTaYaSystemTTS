package com.shotcrete.lotayatts

import android.content.Context
import android.app.ProgressDialog
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.media.MediaPlayer
import android.os.Bundle
import android.os.Environment
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.widget.Button
import android.widget.EditText
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.HashMap
import java.util.Locale
import java.util.UUID
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {
    private var ortEnv: OrtEnvironment? = null
    private var ortSessionMm: OrtSession? = null
    private var progressDialog: ProgressDialog? = null
    
    private var lastAudioFilePath: String? = null
    private var mediaPlayer: MediaPlayer? = null
    private var googleTts: TextToSpeech? = null
    private var isGoogleTtsReady = false

    private val vocabMapMm = mapOf(
        '်' to 0L, 'ာ' to 1L, 'ု' to 2L, 'ိ' to 3L, 'း' to 4L, 'ေ' to 5L, 'သ' to 6L, 'က' to 7L,
        'င' to 8L, 'တ' to 9L, '့' to 10L, 'မ' to 11L, 'ြ' to 12L, 'ည' to 13L, 'ရ' to 14L, 'အ' to 15L,
        'န' to 16L, 'လ' to 17L, 'ှ' to 18L, 'ပ' to 19L, 'စ' to 20L, 'ခ' to 21L, 'ျ' to 22L, 'ူ' to 23L,
        'ွ' to 24L, 'ါ' to 25L, 'ထ' to 26L, 'ဖ' to 27L, 'ံ' to 28L, 'ယ' to 29L, 'ဆ' to 30L, 'ီ' to 31L,
        'ဲ' to 32L, 'ဟ' to 33L, 'ဘ' to 34L, 'ဝ' to 35L, '္' to 36L, 'ဉ' to 37L, 'ဤ' to 38L, 'ဇ' to 39L,
        'ဒ' to 40L, 'ဂ' to 41L, 'ဦ' to 42L, 'ဏ' to 43L, 'ဗ' to 44L, 'ဓ' to 45L, 'ဧ' to 46L, 'ဥ' to 47L,
        'ဩ' to 48L, 'ဌ' to 49L, 'ဋ' to 50L, '\'' to 51L, 'ဣ' to 52L, 'ဍ' to 53L, 'ဿ' to 54L, 'ဈ' to 55L,
        ' ' to 56L
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        googleTts = TextToSpeech(this, this)

        val inputText = findViewById<EditText>(R.id.inputText)
        val speakButton = findViewById<Button>(R.id.speakButton)
        val playLastButton = findViewById<Button>(R.id.playLastButton)
        val pitchSeekBar = findViewById<SeekBar>(R.id.pitchSeekBar)
        val pitchValueText = findViewById<TextView>(R.id.pitchValueText)
        
        // --- Pitch Settings ကို UI ပေါ်တွင် ချိတ်ဆက်မောင်းနှင်ခြင်း ---
        val sharedPref = getSharedPreferences("LoTaYaSettings", Context.MODE_PRIVATE)
        val savedPitch = sharedPref.getFloat("custom_pitch", 1.0f)
        pitchSeekBar.max = 200
        pitchSeekBar.progress = (savedPitch * 100).toInt()
        pitchValueText.text = "အသံအနေအထား: ${savedPitch}x"

        pitchSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                val pitch = progress.toFloat() / 100.0f
                pitchValueText.text = "အသံအနေအထား: ${pitch}x"
                sharedPref.edit().putFloat("custom_pitch", pitch).apply()
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        progressDialog = ProgressDialog(this).apply {
            setMessage("အသံဖိုင်ပြောင်းလဲနေပါသည်... ခဏစောင့်ပါ...")
            setCancelable(false)
        }

        thread(start = true) {
            try {
                ortEnv = OrtEnvironment.getEnvironment()
                val modelFileMm = File(cacheDir, "model.onnx")
                if (!modelFileMm.exists() || modelFileMm.length() < 10_000_000) {
                    if (modelFileMm.exists()) modelFileMm.delete()
                    copyAssetToFile("model.onnx", modelFileMm)
                }
                ortSessionMm = ortEnv?.createSession(modelFileMm.absolutePath)

                runOnUiThread {
                    Toast.makeText(this@MainActivity, "မြန်မာ TTS Engine အဆင်သင့်ဖြစ်ပါပြီ", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                e.printStackTrace()
                runOnUiThread {
                    Toast.makeText(this@MainActivity, "Engine Loading Error: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }

        speakButton.setOnClickListener {
            val rawText = inputText.text.toString().trim()
            if (rawText.isNotEmpty()) {
                val env = ortEnv
                if (ortSessionMm == null || env == null) {
                    Toast.makeText(this, "မြန်မာ Engine အဆင်သင့်မဖြစ်သေးပါ...", Toast.LENGTH_SHORT).show()
                    return@setOnClickListener
                }

                progressDialog?.show()

                thread(start = true) {
                    try {
                        val textSegments = splitByLanguage(rawText)
                        val combinedAudioList = mutableListOf<FloatArray>()
                        val currentPitch = sharedPref.getFloat("custom_pitch", 1.0f)

                        for (segment in textSegments) {
                            if (segment.isEnglish) {
                                if (isGoogleTtsReady) {
                                    val engFloats = generateEnglishAudioFloats(segment.text)
                                    if (engFloats != null && engFloats.isNotEmpty()) {
                                        combinedAudioList.add(engFloats)
                                    }
                                }
                            } else {
                                runMyanmarPipeline(segment.text, env, combinedAudioList, currentPitch)
                            }
                        }

                        if (combinedAudioList.isEmpty()) {
                            runOnUiThread { progressDialog?.dismiss() }
                            return@thread
                        }

                        val totalLength = combinedAudioList.sumOf { it.size }
                        val finalAudioFloats = FloatArray(totalLength)
                        var destPos = 0
                        for (audioChunk in combinedAudioList) {
                            System.arraycopy(audioChunk, 0, finalAudioFloats, destPos, audioChunk.size)
                            destPos += audioChunk.size
                        }

                        var maxVal = 0.0f
                        for (f in finalAudioFloats) {
                            val absF = if (f < 0) -f else f
                            if (absF > maxVal) maxVal = absF
                        }
                        if (maxVal > 0) {
                            val gain = 0.85f / maxVal
                            for (i in finalAudioFloats.indices) {
                                finalAudioFloats[i] = finalAudioFloats[i] * gain
                            }
                        }

                        val sampleRate = 16000
                        runOnUiThread { progressDialog?.dismiss() }

                        val bufferSize = AudioTrack.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT)
                        val audioTrack = AudioTrack(
                            AudioManager.STREAM_MUSIC, 
                            sampleRate, 
                            AudioFormat.CHANNEL_OUT_MONO, 
                            AudioFormat.ENCODING_PCM_FLOAT, 
                            maxOf(bufferSize, finalAudioFloats.size * 4), 
                            AudioTrack.MODE_STATIC 
                        )
                        audioTrack.write(finalAudioFloats, 0, finalAudioFloats.size, AudioTrack.WRITE_BLOCKING)
                        audioTrack.play()

                    } catch (e: Exception) {
                        e.printStackTrace()
                        runOnUiThread {
                            progressDialog?.dismiss()
                            Toast.makeText(this@MainActivity, "Error: ${e.message}", Toast.LENGTH_LONG).show()
                        }
                    }
                }
            } else {
                Toast.makeText(this, "စာသား အရင်ရိုက်ပေးပါ", Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val result = googleTts?.setLanguage(Locale.US)
            isGoogleTtsReady = !(result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED)
        } else {
            isGoogleTtsReady = false
        }
    }

    private fun generateEnglishAudioFloats(text: String): FloatArray? {
        val tempWaveFile = File(cacheDir, "temp_eng_${UUID.randomUUID()}.wav")
        val latch = CountDownLatch(1)
        
        googleTts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) {}
            override fun onDone(utteranceId: String?) { latch.countDown() }
            override fun onError(utteranceId: String?) { latch.countDown() }
        })

        val params = Bundle()
        params.putString(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, "EngWave")
        googleTts?.synthesizeToFile(text, params, tempWaveFile, "EngWave")

        try {
            latch.await(10, TimeUnit.SECONDS)
            if (tempWaveFile.exists() && tempWaveFile.length() > 44) {
                val bytes = tempWaveFile.readBytes()
                tempWaveFile.delete()

                val pcmByteSize = bytes.size - 44
                val shortSamples = pcmByteSize / 2
                val floatArray = FloatArray(shortSamples)
                
                val buffer = ByteBuffer.wrap(bytes, 44, pcmByteSize).order(ByteOrder.LITTLE_ENDIAN)
                for (i in 0 until shortSamples) {
                    if (buffer.hasRemaining()) {
                        floatArray[i] = buffer.short / 32768.0f
                    }
                }
                return downsampleTo16k(floatArray, 24000)
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
        return null
    }

    private fun downsampleTo16k(input: FloatArray, srcSampleRate: Int): FloatArray {
        if (srcSampleRate == 16000) return input
        val ratio = srcSampleRate.toDouble() / 16000.0
        val destLength = (input.size / ratio).toInt()
        val output = FloatArray(destLength)
        for (i in 0 until destLength) {
            val srcIndex = (i * ratio).toInt()
            if (srcIndex < input.size) {
                output[i] = input[srcIndex]
            }
        }
        return output
    }

    private fun runMyanmarPipeline(text: String, env: OrtEnvironment, combinedAudioList: MutableList<FloatArray>, pitch: Float) {
        var processedText = preProcessMyanmarText(text)
        processedText = normalizeNumbers(processedText)

        val cleanChunk = processedText.replace(" ", "")
        val validChars = cleanChunk.filter { vocabMapMm.containsKey(it) }
        if (validChars.length < 2) return

        val tokenList = mutableListOf<Long>()
        tokenList.add(0L)
        for (i in validChars.indices) {
            val id = vocabMapMm[validChars[i]] ?: 56L
            if (id in 0L..56L) {
                tokenList.add(id)
                tokenList.add(0L)
            }
        }

        val inputSequence = tokenList.toLongArray()
        if (inputSequence.isEmpty()) return

        val inputShape = longArrayOf(1, inputSequence.size.toLong())
        val singleShape = longArrayOf(1)
        
        val inputTensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(inputSequence), inputShape)
        val lengthTensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(longArrayOf(inputSequence.size.toLong())), singleShape)
        val scalesTensor = OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(floatArrayOf(0.667f, 1.0f, 0.8f * pitch)), longArrayOf(3))
        
        val attentionMaskSequence = LongArray(inputSequence.size) { 1L }
        val maskTensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(attentionMaskSequence), inputShape)
        val sidTensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(longArrayOf(0L)), singleShape)

        val inputMap = HashMap<String, OnnxTensor>()
        ortSessionMm?.inputNames?.forEach { name ->
            val lowerName = name.lowercase()
            when {
                lowerName.contains("mask") || lowerName.contains("attention") -> inputMap[name] = maskTensor
                name == "input" || lowerName.contains("input_ids") || lowerName == "text" -> inputMap[name] = inputTensor
                lowerName.contains("length") -> inputMap[name] = lengthTensor
                lowerName.contains("scale") -> inputMap[name] = scalesTensor
                lowerName.contains("sid") || lowerName.contains("speaker") -> inputMap[name] = sidTensor
            }
        }

        val results = ortSessionMm?.run(inputMap)
        val outputTensor = results?.get(0) as? OnnxTensor
        outputTensor?.let {
            val floatBuffer = it.floatBuffer
            val audioFloats = FloatArray(floatBuffer.remaining())
            floatBuffer.get(audioFloats)
            combinedAudioList.add(audioFloats)
        }
        inputTensor.close()
        lengthTensor.close()
        scalesTensor.close()
        maskTensor.close()
        sidTensor.close()
        results?.close()
    }

    data class LangSegment(val text: String, val isEnglish: Boolean)
    
    private fun splitByLanguage(text: String): List<LangSegment> {
        val segments = mutableListOf<LangSegment>()
        if (text.isEmpty()) return segments
        var currentSegment = StringBuilder()
        var currentIsEnglish: Boolean? = null

        for (ch in text) {
            val computedIsEnglish = if (ch in " .,!?'-" && currentIsEnglish != null) {
                currentIsEnglish!! 
            } else {
                ch in 'a'..'z' || ch in 'A'..'Z'
            }

            if (currentIsEnglish == null) {
                currentIsEnglish = computedIsEnglish
                currentSegment.append(ch)
            } else if (currentIsEnglish == computedIsEnglish) {
                currentSegment.append(ch)
            } else {
                if (currentSegment.toString().trim().isNotEmpty()) {
                    segments.add(LangSegment(currentSegment.toString(), currentIsEnglish!!))
                }
                currentSegment = StringBuilder().append(ch)
                currentIsEnglish = computedIsEnglish
            }
        }
        if (currentSegment.toString().trim().isNotEmpty()) {
            segments.add(LangSegment(currentSegment.toString(), currentIsEnglish ?: false))
        }
        return segments
    }

    private fun preProcessMyanmarText(text: String): String {
        var res = text
        res = res.replace(Regex("[xX._*#+=()_\\-]"), "")
        res = res.replace("ဪ", "အော်")
        res = res.replace("၎င်း", "လဂေါင်း")
        res = res.replace("ဖြစ်၏", "ဖြစ်အီ")
        res = res.replace("ဗျ", "ဗျ") 
        res = res.replace("၏", "အီး")
        res = res.replace("၌", "နှိုက်")
        res = res.replace("၍", "ရွေ့")
        res = res.replace("ဤ", "အီ")
        res = res.replace("ဘဏ္ဍာ", "ဘန်ဒါ")
        res = res.replace("သဏ္ဌာန်", "သန်ထန်")
        if (!res.endsWith(" ")) res = "$res   "
        return res
    }

    private fun normalizeNumbers(text: String): String {
        var res = text
        val numMap = mapOf(
            "0" to "သုည", "1" to "တစ်", "2" to "နှစ်", "3" to "သုံး", "4" to "လေး", "5" to "ငါး",
            "6" to "ခြောက်", "7" to "ခုနစ်", "8" to "ရှစ်", "9" to "ကိုး",
            "၀" to "သုည", "၁" to "တစ်", "၂" to "နှစ်", "၃" to "သုံး", "၄" to "လေး", "၅" to "ငါး",
            "၆" to "ခြောက်", "၇" to "ခုနစ်", "၈" to "ရှစ်", "၉" to "ကိုး"
        )
        for ((num, txt) in numMap) { res = res.replace(num, txt) }
        return res
    }

    private fun copyAssetToFile(assetName: String, outFile: File) {
        assets.open(assetName).use { inputStream ->
            FileOutputStream(outFile).use { outputStream ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
                outputStream.flush()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        googleTts?.stop()
        googleTts?.shutdown()
        mediaPlayer?.release()
        progressDialog?.dismiss()
        ortSessionMm?.close()
        ortEnv?.close()
    }
}
