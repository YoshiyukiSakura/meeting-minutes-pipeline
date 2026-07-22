import AVFoundation
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers
import Vision

struct FrameRecord: Codable {
    let time: Double
    let actualTime: Double
    let path: String
}

struct OCRRegion: Codable {
    let label: String
    let box: [Double]
}

struct OCRRegionsRecord: Codable {
    let time: Double
    let actualTime: Double?
    let path: String
    let regions: [OCRRegion]
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(1)
}

func printJSON(_ value: Any) {
    do {
        let data = try JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys])
        FileHandle.standardOutput.write(data)
        FileHandle.standardOutput.write("\n".data(using: .utf8)!)
    } catch {
        fail("JSON serialization failed: \(error)")
    }
}

func seconds(_ time: CMTime) -> Double {
    let value = CMTimeGetSeconds(time)
    return value.isFinite ? value : 0
}

func probe(_ args: [String]) {
    guard args.count == 1 else { fail("usage: probe <input>") }
    let url = URL(fileURLWithPath: args[0])
    let asset = AVURLAsset(url: url)
    var tracks: [[String: Any]] = []
    for track in asset.tracks {
        let transformed = track.naturalSize.applying(track.preferredTransform)
        tracks.append([
            "mediaType": track.mediaType.rawValue,
            "duration": seconds(track.timeRange.duration),
            "width": abs(Double(transformed.width)),
            "height": abs(Double(transformed.height)),
            "nominalFrameRate": Double(track.nominalFrameRate),
            "estimatedDataRate": Double(track.estimatedDataRate),
        ])
    }
    printJSON([
        "duration": seconds(asset.duration),
        "tracks": tracks,
    ])
}

func writeJPEG(_ image: CGImage, to url: URL) throws {
    guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.jpeg.identifier as CFString, 1, nil) else {
        throw NSError(domain: "macos_media", code: 1, userInfo: [NSLocalizedDescriptionKey: "Cannot create image destination"])
    }
    CGImageDestinationAddImage(destination, image, [kCGImageDestinationLossyCompressionQuality: 0.86] as CFDictionary)
    if !CGImageDestinationFinalize(destination) {
        throw NSError(domain: "macos_media", code: 2, userInfo: [NSLocalizedDescriptionKey: "Cannot finalize JPEG"])
    }
}

func frames(_ args: [String]) {
    guard args.count >= 3 else { fail("usage: frames <input> <times_json> <outdir> [max_width]") }
    let input = URL(fileURLWithPath: args[0])
    let timesURL = URL(fileURLWithPath: args[1])
    let outDir = URL(fileURLWithPath: args[2], isDirectory: true)
    let maxWidth = args.count >= 4 ? Double(args[3]) ?? 1280 : 1280

    guard let timesData = try? Data(contentsOf: timesURL),
          let times = try? JSONDecoder().decode([Double].self, from: timesData) else {
        fail("Cannot decode times JSON")
    }
    try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

    let asset = AVURLAsset(url: input)
    let generator = AVAssetImageGenerator(asset: asset)
    generator.appliesPreferredTrackTransform = true
    generator.requestedTimeToleranceBefore = CMTime(seconds: 0.1, preferredTimescale: 600)
    generator.requestedTimeToleranceAfter = CMTime(seconds: 0.1, preferredTimescale: 600)
    generator.maximumSize = CGSize(width: maxWidth, height: maxWidth * 2)

    var output: [[String: Any]] = []
    for requested in times {
        let time = CMTime(seconds: requested, preferredTimescale: 600)
        do {
            var actual = CMTime.zero
            let image = try generator.copyCGImage(at: time, actualTime: &actual)
            let stem = String(format: "frame_%08.3f", requested).replacingOccurrences(of: ".", with: "_")
            let fileName = stem + ".jpg"
            let path = outDir.appendingPathComponent(fileName)
            try writeJPEG(image, to: path)
            output.append([
                "time": requested,
                "actualTime": seconds(actual),
                "path": path.path,
            ])
        } catch {
            output.append([
                "time": requested,
                "error": "\(error)",
            ])
        }
    }
    printJSON(output)
}

func ocr(_ args: [String]) {
    guard args.count == 1 else { fail("usage: ocr <frames_manifest_json>") }
    let manifestURL = URL(fileURLWithPath: args[0])
    guard let data = try? Data(contentsOf: manifestURL),
          let records = try? JSONDecoder().decode([FrameRecord].self, from: data) else {
        fail("Cannot decode frames manifest")
    }

    var output: [[String: Any]] = []
    for record in records {
        let url = URL(fileURLWithPath: record.path)
        guard let cgSource = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(cgSource, 0, nil) else {
            output.append(["time": record.time, "path": record.path, "texts": [], "error": "cannot_read_image"])
            continue
        }

        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = true
        request.recognitionLanguages = ["zh-Hans", "zh-Hant", "en-US", "ja-JP"]
        let handler = VNImageRequestHandler(cgImage: image, options: [:])
        do {
            try handler.perform([request])
            var texts: [[String: Any]] = []
            for observation in request.results ?? [] {
                guard let candidate = observation.topCandidates(1).first else { continue }
                let box = observation.boundingBox
                texts.append([
                    "text": candidate.string,
                    "confidence": Double(candidate.confidence),
                    "bbox": [
                        "x": Double(box.origin.x),
                        "y": Double(box.origin.y),
                        "width": Double(box.size.width),
                        "height": Double(box.size.height),
                    ],
                ])
            }
            output.append(["time": record.time, "path": record.path, "texts": texts])
        } catch {
            output.append(["time": record.time, "path": record.path, "texts": [], "error": "\(error)"])
        }
    }
    printJSON(output)
}

func recognizeTexts(_ image: CGImage) throws -> [[String: Any]] {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true
    request.recognitionLanguages = ["zh-Hans", "zh-Hant", "en-US", "ja-JP"]
    let handler = VNImageRequestHandler(cgImage: image, options: [:])
    try handler.perform([request])
    var texts: [[String: Any]] = []
    for observation in request.results ?? [] {
        guard let candidate = observation.topCandidates(1).first else { continue }
        let box = observation.boundingBox
        texts.append([
            "text": candidate.string,
            "confidence": Double(candidate.confidence),
            "bbox": [
                "x": Double(box.origin.x),
                "y": Double(box.origin.y),
                "width": Double(box.size.width),
                "height": Double(box.size.height),
            ],
        ])
    }
    return texts
}

func upscaleForOCR(_ image: CGImage, factor: Int = 4) -> CGImage? {
    let width = image.width * factor
    let height = image.height * factor
    let colorSpace = image.colorSpace ?? CGColorSpaceCreateDeviceRGB()
    guard let context = CGContext(
        data: nil,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: 0,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else {
        return nil
    }
    context.interpolationQuality = .high
    context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
    return context.makeImage()
}

func ocrRegions(_ args: [String]) {
    guard args.count == 1 else { fail("usage: ocr-regions <regions_manifest_json>") }
    let manifestURL = URL(fileURLWithPath: args[0])
    guard let data = try? Data(contentsOf: manifestURL),
          let records = try? JSONDecoder().decode([OCRRegionsRecord].self, from: data) else {
        fail("Cannot decode OCR regions manifest")
    }

    var output: [[String: Any]] = []
    for record in records {
        let url = URL(fileURLWithPath: record.path)
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            output.append(["time": record.time, "path": record.path, "regions": [], "error": "cannot_read_image"])
            continue
        }
        var regionOutput: [[String: Any]] = []
        for region in record.regions {
            guard region.box.count == 4 else {
                regionOutput.append(["label": region.label, "texts": [], "error": "invalid_box"])
                continue
            }
            let x1 = max(0, min(1, region.box[0]))
            let y1 = max(0, min(1, region.box[1]))
            let x2 = max(0, min(1, region.box[2]))
            let y2 = max(0, min(1, region.box[3]))
            guard x2 > x1, y2 > y1 else {
                regionOutput.append(["label": region.label, "texts": [], "error": "invalid_box"])
                continue
            }
            let rect = CGRect(
                x: CGFloat(x1) * CGFloat(image.width),
                y: CGFloat(y1) * CGFloat(image.height),
                width: CGFloat(x2 - x1) * CGFloat(image.width),
                height: CGFloat(y2 - y1) * CGFloat(image.height)
            ).integral
            guard let crop = image.cropping(to: rect) else {
                regionOutput.append(["label": region.label, "texts": [], "error": "cannot_crop"])
                continue
            }
            do {
                let ocrImage = upscaleForOCR(crop) ?? crop
                regionOutput.append(["label": region.label, "texts": try recognizeTexts(ocrImage)])
            } catch {
                regionOutput.append(["label": region.label, "texts": [], "error": "\(error)"])
            }
        }
        var item: [String: Any] = ["time": record.time, "path": record.path, "regions": regionOutput]
        if let actualTime = record.actualTime {
            item["actualTime"] = actualTime
        }
        output.append(item)
    }
    printJSON(output)
}

let argv = Array(CommandLine.arguments.dropFirst())
guard let command = argv.first else {
    fail("usage: macos_media.swift <probe|frames|ocr|ocr-regions> ...")
}
let rest = Array(argv.dropFirst())
switch command {
case "probe":
    probe(rest)
case "frames":
    frames(rest)
case "ocr":
    ocr(rest)
case "ocr-regions":
    ocrRegions(rest)
default:
    fail("unknown command: \(command)")
}
