import UIKit
import CoreML
import Vision

class ViewController: UIViewController {

    private var faceRect = CGRect.zero
    @IBOutlet weak var imageView: UIImageView!
    var faceImage:UIImage?
    
    private var image: UIImage? {
        didSet {
            DispatchQueue.main.async {
                self.imageView.image = self.image
            }
        }
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        faceImage = imageView.image
    }
    
    @IBAction func detectFaces(_ sender: Any) {
        detectFaces(within: imageView.image!)
    }
    
    @IBAction func recognizeFaces(_ sender: Any) {
        analyze()
    }
    
    func detectFaces(within uiImage: UIImage) {
        guard let cgImage = uiImage.cgImage else {
            fatalError("Invalid image")
        }
        let request = VNDetectFaceRectanglesRequest { [unowned self] request, err in
            guard err == nil else {
                print(err!)
                return
            }
            
            guard let results = request.results else {
                return
            }
            
            for case let result as VNFaceObservation in results {
                self.image = self.imageWith(size: uiImage.size, style: { ctx in
                    self.faceRect = self.rect(fromRelative: result.boundingBox, size: uiImage.size)
                    ctx.setStrokeColor(UIColor.red.cgColor)
                    ctx.stroke(self.faceRect, width: 12.0)
                })!
            }
        }
        request.preferBackgroundProcessing = true
        DispatchQueue.global(qos: DispatchQoS.QoSClass.userInitiated).async {
            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            try? handler.perform([request])
        }
    }
    
    // MARK - Private
    
    private func imageWith(size: CGSize, style: (CGContext) -> Void) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, false, 0.0)
        defer { UIGraphicsEndImageContext() }
        guard let ctx = UIGraphicsGetCurrentContext() else {
            return nil
        }
        self.image?.draw(at: .zero)
        style(ctx)
        return UIGraphicsGetImageFromCurrentImageContext()
    }
    
    private func rect(fromRelative boundingBox: CGRect, size: CGSize) -> CGRect {
        var rect = CGRect(
            x: boundingBox.origin.x * size.width,
            y: size.height - boundingBox.origin.y * size.height,
            width: boundingBox.width * size.width,
            height: boundingBox.height * size.height
        )
        rect.origin.y -= rect.height
        return rect
    }
    
    func analyze() {
        DispatchQueue.global().async {
            self.highlightFaces(for: self.faceImage!) { (resultImage) in
                DispatchQueue.main.async {
                    self.imageView.image = resultImage
                }
            }
        }
    }
    
    open func highlightFaces(for source: UIImage, complete: @escaping (UIImage) -> Void) {
        var resultImage = source
        let detectFaceRequest = VNDetectFaceLandmarksRequest { (request, error) in
            if error == nil {
                if let results = request.results as? [VNFaceObservation] {
                    print("Found \(results.count) faces")
                    
                    for faceObservation in results {
                        guard let landmarks = faceObservation.landmarks else {
                            continue
                        }
                        let boundingRect = faceObservation.boundingBox
                        var landmarkRegions: [VNFaceLandmarkRegion2D] = []
                        if let faceContour = landmarks.faceContour {
                            landmarkRegions.append(faceContour)
                        }
                        if let leftEye = landmarks.leftEye {
                            landmarkRegions.append(leftEye)
                        }
                        if let rightEye = landmarks.rightEye {
                            landmarkRegions.append(rightEye)
                        }
                        if let nose = landmarks.nose {
                            landmarkRegions.append(nose)
                        }
                        if let noseCrest = landmarks.noseCrest {
                            landmarkRegions.append(noseCrest)
                        }
                        if let medianLine = landmarks.medianLine {
                            landmarkRegions.append(medianLine)
                        }
                        if let outerLips = landmarks.outerLips {
                            landmarkRegions.append(outerLips)
                        }
                        
                        if let leftEyebrow = landmarks.leftEyebrow {
                            landmarkRegions.append(leftEyebrow)
                        }
                        if let rightEyebrow = landmarks.rightEyebrow {
                            landmarkRegions.append(rightEyebrow)
                        }
                        
                        if let innerLips = landmarks.innerLips {
                            landmarkRegions.append(innerLips)
                        }
                        if let leftPupil = landmarks.leftPupil {
                            landmarkRegions.append(leftPupil)
                        }
                        if let rightPupil = landmarks.rightPupil {
                            landmarkRegions.append(rightPupil)
                        }
                        
                        resultImage = self.drawOnImage(source: resultImage,
                                                       boundingRect: boundingRect,
                                                       faceLandmarkRegions: landmarkRegions)
                        
                        
                    }
                }
            } else {
                print(error!.localizedDescription)
            }
            complete(resultImage)
        }
        
        let vnImage = VNImageRequestHandler(cgImage: source.cgImage!, options: [:])
        try? vnImage.perform([detectFaceRequest])
    }

    fileprivate func drawOnImage(source: UIImage,
                                 boundingRect: CGRect,
                                 faceLandmarkRegions: [VNFaceLandmarkRegion2D]) -> UIImage {
        UIGraphicsBeginImageContextWithOptions(source.size, false, 1)
        let context = UIGraphicsGetCurrentContext()!
        context.translateBy(x: 0, y: source.size.height)
        context.scaleBy(x: 1.0, y: -1.0)
        context.setBlendMode(CGBlendMode.colorBurn)
        context.setLineJoin(.round)
        context.setLineCap(.round)
        context.setShouldAntialias(true)
        context.setAllowsAntialiasing(true)
        
        let rectWidth = source.size.width * boundingRect.size.width
        let rectHeight = source.size.height * boundingRect.size.height
        
        //draw image
        let rect = CGRect(x: 0, y:0, width: source.size.width, height: source.size.height)
        context.draw(source.cgImage!, in: rect)
        
        
        //draw bound rect
        var fillColor = UIColor.red
        fillColor.setFill()
        context.addRect(CGRect(x: boundingRect.origin.x * source.size.width, y:boundingRect.origin.y * source.size.height, width: rectWidth, height: rectHeight))
        context.drawPath(using: CGPathDrawingMode.stroke)
        
        //draw overlay
        fillColor = UIColor.red
        fillColor.setStroke()
        context.setLineWidth(4.0)
        for faceLandmarkRegion in faceLandmarkRegions {
            var points: [CGPoint] = []
            for i in 0..<faceLandmarkRegion.pointCount {
                let point = faceLandmarkRegion.normalizedPoints[i]
                let p = CGPoint(x: CGFloat(point.x), y: CGFloat(point.y))
                points.append(p)
            }
            let mappedPoints = points.map { CGPoint(x: boundingRect.origin.x * source.size.width + $0.x * rectWidth, y: boundingRect.origin.y * source.size.height + $0.y * rectHeight) }
            context.addLines(between: mappedPoints)
            context.drawPath(using: CGPathDrawingMode.stroke)
        }
        
        let coloredImg : UIImage = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()
        return coloredImg
    }

}

